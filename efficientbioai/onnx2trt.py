import onnx
import pycuda.autoinit  # noqa F401
import tensorrt as trt
import json
import pycuda.driver as cuda  # noqa F401
import os


def onnx2trt(
    onnx_model,
    trt_path,
    log_level=trt.Logger.ERROR,
    max_workspace_size=1 << 30,
    mode="fp32",
    dynamic_range_file=None,
    input_names=None,
    input_size=None,
    dynamic_batch=None,
):
    if os.path.exists(trt_path):
        print(f'The "{trt_path}" exists. Remove it and continue.')
        os.remove(trt_path)

    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    )  # noqa: E501
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ""
        for error in range(parser.num_errors):
            error_msgs += f"{parser.get_error(error)}\n"
        raise RuntimeError(f"parse onnx failed:\n{error_msgs}")

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if dynamic_range_file:
            with open(dynamic_range_file, "r") as f:
                dynamic_range = json.load(f)["tensorrt"]["blob_range"]

            for input_index in range(network.num_inputs):
                input_tensor = network.get_input(input_index)
                if input_tensor.name in dynamic_range:
                    amax = dynamic_range[input_tensor.name]
                    input_tensor.dynamic_range = (-amax, amax)
                    print(
                        f"Set dynamic range of {input_tensor.name} as [{-amax}, {amax}]"
                    )

            for layer_index in range(network.num_layers):
                layer = network[layer_index]
                output_tensor = layer.get_output(0)
                if output_tensor.name in dynamic_range:
                    amax = dynamic_range[output_tensor.name]
                    output_tensor.dynamic_range = (-amax, amax)
                    print(
                        f"Set dynamic range of {output_tensor.name} as [{-amax}, {amax}]"
                    )
        else:
            pass

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_names[0],
        tuple([dynamic_batch[0], *input_size]),
        tuple([dynamic_batch[1], *input_size]),
        tuple([dynamic_batch[2], *input_size]),
    )  # for the moment, only suppport 1 input senario.
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)

    with open(trt_path, mode="wb") as f:
        f.write(bytearray(engine.serialize()))
    return engine
