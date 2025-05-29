from src.format_converter import FormatConverter

def main():
    converter = FormatConverter()
    input_file = "script/程聿怀男_本1_文字版.docx"
    output_file = "script/程聿怀男_本1_文字版.md"
    
    try:
        result = converter.process_document(input_file, output_file)
        print(f"转换成功！输出文件：{result}")
    except Exception as e:
        print(f"转换失败：{str(e)}")

if __name__ == "__main__":
    main() 