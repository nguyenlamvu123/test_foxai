import gradio as gr
from coordinate import *

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Tải ảnh đầu vào", type="filepath")
        with gr.Column():
            th1 = gr.Number(label="Threshold1 cho Canny", value=50)
            th2 = gr.Number(label="Threshold2 cho Canny", value=90)
            process_btn = gr.Button("Xử lý")

    # with gr.Row():
    #     output_images = gr.Gallery(label="Kết quả xử lý", show_label=True, columns=3)

    with gr.Row():
        checkbox = gr.CheckboxGroup(
            label="Chọn các ảnh muốn xử lý",
            choices=list_image_filenames()
        )

    with gr.Row():
        image_gallery = gr.Gallery(label="Xem trước ảnh đã chọn", columns=3)

    with gr.Row():
        analyze_btn = gr.Button("Phân tích các ảnh đã chọn")
        text_out = gr.Textbox(label="Kết quả tóm tắt")
        json_out = gr.JSON(label="Kết quả chi tiết")

    # Sự kiện xử lý đầu vào
    def run_pipeline(img, th1, th2):
        img = cv2.imread(img, flags=0)
        findcoord(img, th1, th2)

        # output_images = []
        # for file in os.listdir():
        #     if file.endswith(extout) and file.startswith(imgout):
        #         img_out = cv2.imread(file)
        #         img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        #         output_images.append(img_rgb)
        # return output_images

    process_btn.click(
        fn=run_pipeline,
        inputs=(input_img, th1, th2),
        # outputs=output_images
        )

    # Khi người dùng nhấn vào ảnh gallery, truyền ảnh sang ô "selected"
    def handle_selection(evt: gr.SelectData):
        # evt.value là ảnh (NumPy RGB) nếu ảnh trong Gallery là dạng mảng
        if evt.value is None:
            return None  # tránh lỗi nếu không có ảnh
        path = evt.value['image']['path']
        img_out = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        return img_rgb

    checkbox.change(fn=show_selected_images, inputs=checkbox, outputs=image_gallery)
    # output_images.select(fn=handle_selection, inputs=None, outputs=selected)
    # # Khi người dùng nhấn nút phân tích
    analyze_btn.click(fn=analyze_images, inputs=checkbox, outputs=[text_out, json_out])

demo.launch()
