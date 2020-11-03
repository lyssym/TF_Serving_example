# _*_ coding: utf-8 _*_

import os
import shutil
import os.path as osp
import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.python.tools import import_pb_to_tensorboard
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from keras import backend as K

import bilsm_crf_model


# h5文件 生成 pb冻结文件
def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir, model_name), output_dir)


# pb冻结文件生成 pb签名文件
def save_server_graph(graph_pb, h5_model, builder):
    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        sigs = {}
        with tf.Session(graph=tf.Graph()) as sess:
            tf.import_graph_def(graph_def, name="")
            g = tf.get_default_graph()
            inp = g.get_tensor_by_name(h5_model.input.name)
            out = g.get_tensor_by_name(h5_model.output.name)

            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"inputs": inp}, {"outputs": out})

            builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                                 signature_def_map=sigs)
    builder.save()


# h5文件 生成 pb签名文件
def generate_server_h52pb(weight_file, weight_dir, output_dir, export_dir):
    h5_model, _ = bilsm_crf_model.create_model(train=False)
    weight_file_path = osp.join(weight_dir, weight_file)
    h5_model.load_weights(weight_file_path)
    output_graph_name = weight_file[:-3] + '.pb'
    h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)

    if osp.exists(export_dir):
        shutil.rmtree(export_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    graph_pb = osp.join(output_dir, output_graph_name)
    save_server_graph(graph_pb, h5_model, builder)


# h5文件 生成 pb签名文件， 无tensorboard文件
def generate_server_h52pb_simple(weight_file, weight_dir, export_dir):
    h5_model, _ = bilsm_crf_model.create_model(train=False)
    weight_file_path = osp.join(weight_dir, weight_file)
    h5_model.load_weights(weight_file_path)

    if osp.exists(export_dir):
        shutil.rmtree(export_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input': h5_model.input}, outputs={'output': h5_model.output})

    builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature
            })
    builder.save()


# checkpoint文件 生成 pb签名文件
def generate_server_ckpt2pb(input_checkpoint, export_path):
    if osp.exists(export_path):
        shutil.rmtree(export_path)

    graph = tf.Graph()
    with graph.as_default():
        sigs = {}
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
            saver.restore(sess, input_checkpoint)                # 恢复图并得到数据

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]

            argMax = graph.get_operation_by_name("score/ArgMax").outputs[0]

            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
                tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs={"input_x": input_x,
                            "keep_prob": keep_prob},
                    outputs={"argMax": argMax})

            builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                                 signature_def_map=sigs)
            builder.save()


if __name__ == "__main__":
    weight_file = "crf.h5"
    weight_dir = "model"
    output_dir = "trans"
    export_dir = "pb_model"

    # generate_server_h52pb(weight_file=weight_file, weight_dir=weight_dir,
    #                       output_dir=output_dir, export_dir=export_dir)

    generate_server_h52pb_simple(weight_file=weight_file, weight_dir=weight_dir,
                                 export_dir=export_dir)

    # input_checkpoint = "textcnn/best_validation"
    # export_path = "text_cnn"
    # generate_server_ckpt2pb(input_checkpoint, export_path=export_path)
