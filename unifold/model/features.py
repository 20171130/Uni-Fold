# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code to generate processed features."""
import copy
from typing import List, Mapping, Tuple
from unifold.model.tf import input_pipeline
from unifold.model.tf import proteins_dataset
from unifold.model import all_atom
import ml_collections
import numpy as np
import tensorflow.compat.v1 as tf
from unifold.common import residue_constants
from unifold.model import quat_affine
from jax import numpy as jnp

FeatureDict = Mapping[str, np.ndarray]


def make_data_config(
    config: ml_collections.ConfigDict,
    num_res: int,
    ) -> Tuple[ml_collections.ConfigDict, List[str]]:
  """Makes a data config for the input pipeline."""
  cfg = copy.deepcopy(config.data)

  feature_names = cfg.common.unsupervised_features
  if cfg.common.use_templates:
    feature_names += cfg.common.template_features

  with cfg.unlocked():
    cfg.eval.crop_size = num_res

  return cfg, feature_names


def tf_example_to_features(tf_example: tf.train.Example,
                           config: ml_collections.ConfigDict,
                           random_seed: int = 0) -> FeatureDict:
  """Converts tf_example to numpy feature dictionary."""
  num_res = int(tf_example.features.feature['seq_length'].int64_list.value[0])
  cfg, feature_names = make_data_config(config, num_res=num_res)

  if 'deletion_matrix_int' in set(tf_example.features.feature):
    deletion_matrix_int = (
        tf_example.features.feature['deletion_matrix_int'].int64_list.value)
    feat = tf.train.Feature(float_list=tf.train.FloatList(
        value=map(float, deletion_matrix_int)))
    tf_example.features.feature['deletion_matrix'].CopyFrom(feat)
    del tf_example.features.feature['deletion_matrix_int']

  tf_graph = tf.Graph()
  with tf_graph.as_default(), tf.device('/device:CPU:0'):
    tf.compat.v1.set_random_seed(random_seed)
    tensor_dict = proteins_dataset.create_tensor_dict(
        raw_data=tf_example.SerializeToString(),
        features=feature_names)
    processed_batch = input_pipeline.process_tensors_from_config(
        tensor_dict, cfg)

  tf_graph.finalize()

  with tf.Session(graph=tf_graph) as sess:
    features = sess.run(processed_batch)

  return {k: v for k, v in features.items() if v.dtype != 'O'}


def np_example_to_features(np_example: FeatureDict,
                           config: ml_collections.ConfigDict,
                           random_seed: int = 0) -> FeatureDict:
  """Preprocesses NumPy feature dict using TF pipeline."""
  np_example = dict(np_example)
  num_res = int(np_example['seq_length'][0])
  cfg, feature_names = make_data_config(config, num_res=num_res)

  if 'deletion_matrix_int' in np_example:
    np_example['deletion_matrix'] = (
        np_example.pop('deletion_matrix_int').astype(np.float32))

  tf_graph = tf.Graph()
  with tf_graph.as_default(), tf.device('/device:CPU:0'):
    tf.compat.v1.set_random_seed(random_seed)
    tensor_dict = proteins_dataset.np_to_tensor_dict(
        np_example=np_example, features=feature_names)

    processed_batch = input_pipeline.process_tensors_from_config(
        tensor_dict, cfg)

  tf_graph.finalize()

  with tf.Session(graph=tf_graph) as sess:
    features = sess.run(processed_batch)
    
  dic = {k: v for k, v in features.items() if v.dtype != 'O'}
  if 'template_aatype' in dic:
    # torsion angles
    batch = dic
    rets = []  
    for i in range(batch['template_aatype'].shape[0]):
      # for each template
      ret = all_atom.atom37_to_torsion_angles(
          aatype=batch['template_aatype'][i],
          all_atom_pos=batch['template_all_atom_positions'][i],
          all_atom_mask=batch['template_all_atom_masks'][i],
          # Ensure consistent behaviour during testing:
          placeholder_for_undefined=not config.model.global_config.zero_init)
      rets.append(ret)
    rets = {k:np.stack([item[k] for item in rets]) for k in rets[0].keys()}
    dic.update(rets)
      
    dic['template_point'] = []
    dic['template_affine_vec'] = []
    for i in range(batch['template_all_atom_positions'].shape[0]):
      # template pair
      dic['template_point'].append([])
      dic['template_affine_vec'].append([])
      for j in range(batch['template_all_atom_positions'].shape[1]):
        # four templates
        tmp = batch['template_all_atom_positions'][i,j]
        n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
        rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=tmp[:, n],
            ca_xyz=tmp[:, ca],
            c_xyz=tmp[:, c])
        affines = quat_affine.QuatAffine(
            quaternion=quat_affine.rot_to_quat(rot, unstack_inputs=True),
            translation=trans,
            rotation=rot,
            unstack_inputs=True)
        point = jnp.stack([jnp.expand_dims(x, axis=-2) for x in affines.translation])
        affine_vec = jnp.stack(affines.invert_point(point, extra_dims=1))
        
        dic['template_point'][i].append(point)
        dic['template_affine_vec'][i].append(point)
      dic['template_point'][i] = jnp.stack(dic['template_point'][i])
      dic['template_affine_vec'][i] = jnp.stack(dic['template_affine_vec'][i])
    dic['template_point'] = jnp.stack(dic['template_point'])
    dic['template_affine_vec'] = jnp.stack(dic['template_affine_vec'])

    # shape (2, 4, 3, 1, 64) (2, 4, 3, 64, 64)
    
  return dic
