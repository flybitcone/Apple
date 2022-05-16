##
import tensorflow as tf
##
# 转化模型为tflite格式
model = tf.keras.models.load_model('apple_test0731.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存模型
with open('apple_model_07.31.tflite', 'wb') as f:
  f.write(tflite_model)

##测试tflite文件
interpreter = tf.lite.Interpreter(model_path="apple_model_07.31.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
