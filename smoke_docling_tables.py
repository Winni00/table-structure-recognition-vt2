from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

base_dir = Path("/cluster/home/trinhwin/vt2/docling")
input_pdf = base_dir / "data" / "10.1007%s11250-017-1251-6.pdf"
output_dir = base_dir / "results" / "smoke_test"
output_dir.mkdir(parents=True, exist_ok=True)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.table_structure_options.do_cell_matching = True
pipeline_options.accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

result = converter.convert(str(input_pdf))
doc = result.document

doc.save_as_markdown(output_dir / "output.md")
doc.save_as_json(output_dir / "output.json")

print("Done")
print(f"Saved results to: {output_dir}")
