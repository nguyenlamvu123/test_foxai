# test_app.py
import os, json, unittest
from fastapi.testclient import TestClient
from api_be import app
from coordinate import expectedimg, expected_output_img, expected_jso

class TestFastAPI(unittest.TestCase):
    def setUp(self):
        for c in os.listdir():
            if c.endswith('.csv'):
                os.remove(c)  # xóa hết các file csv đầu ra trước khi chạy test
        self.client = TestClient(app)

    def test_upload_pdf(self):
        response = self.client.post(
            "/upload/",
            files={
                "file": (expectedimg, open(expectedimg, 'rb'), 'image/png'),
            },
            data={
                'th1': '50',
                'th2': '90',
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": f"Tài liệu '{expectedimg}' đã được xử lý thành công."}
        )

        for out_im in expected_output_img:
            self.assertTrue(os.path.isfile(out_im))
            os.remove(out_im)

        csv = [c for c in os.listdir() if c.endswith('.csv')]
        assert len(csv) == 1
        csv = csv[0]
        with open(csv, 'r', encoding='utf-8') as file:
            jso = json.load(file)
        self.assertEqual(jso, expected_jso)
        os.remove(csv)

if __name__ == "__main__":
    unittest.main()
