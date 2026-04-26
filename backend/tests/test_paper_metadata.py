from app.services.mineru import ParseResult
from app.services.paper_metadata import extract_paper_title, filename_to_title


def test_extract_title_from_mineru_first_page_header_block():
    parsed = ParseResult(
        markdown_path="paper.md",
        markdown_content="",
        content_list_path="paper_content_list.json",
        content_list=[
            {
                "type": "text",
                "page_idx": 0,
                "text": """Nature Machine Intelligence | Volume 7 | February 2025 | 270-277
270
nature machine intelligence
Article
https://doi.org/10.1038/s42256-024-00972-x
Battery lifetime prediction across diverse
ageing conditions with inter-cell
deep learning

Han Zhang1,2,4, Yuqi Li1,3,4, Shun Zheng1, Ziheng Lu1
Accurately predicting battery lifetime in early cycles holds tremendous value in real-world applications.""",
            }
        ],
        output_dir=".",
        assets_dir=".",
        page_count=1,
    )

    assert (
        extract_paper_title(parsed, filename="s42256-024-00972-x.pdf")
        == "Battery lifetime prediction across diverse ageing conditions with inter-cell deep learning"
    )


def test_filename_title_fallback_is_readable():
    assert filename_to_title("my_uploaded-paper_v2.pdf") == "my uploaded paper v2"


if __name__ == "__main__":
    test_extract_title_from_mineru_first_page_header_block()
    test_filename_title_fallback_is_readable()
    print("test_paper_metadata.py: ok")
