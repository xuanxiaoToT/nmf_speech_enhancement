"""
这是整个的降噪系统框架的实现，包含VAD，噪声字典更新机制。
VAD使用DNN版本与目标频段比值两种方法测试。
"""
from stream import StreamBuilder


def main():
    builder = StreamBuilder(stream_type='file', filename='speech/train/fadg0_sa1.wav')
    stream = builder.get_instance()
    print(len(stream.next_frame()))


if __name__ == '__main__':
    main()
