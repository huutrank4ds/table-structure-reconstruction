
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pymupdf

from document_processing.schemas.document import RasterizedDocument, RasterizedPage


type PDFSource = str | Path | bytes


class DocumentRasterizer:
    def __init__(self, dpi: int = 300):
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        self.dpi = dpi

    @staticmethod
    def _open_pdf(source: PDFSource) -> pymupdf.Document:
        if isinstance(source, bytes):
            if not source:
                raise ValueError("Empty PDF bytes")
            document = pymupdf.open(stream=source, filetype="pdf")
        else:
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"PDF not found: {path}")
            document = pymupdf.open(str(path))

        if not document.is_pdf or document.needs_pass:
            document.close()
            raise ValueError("Invalid or password-protected PDF")

        return document

    @staticmethod
    def _validate_pages(page_ids: Sequence[int], page_count: int) -> None:
        if any(type(i) is not int or not 0 <= i < page_count for i in page_ids):
            raise ValueError(f"page_ids must be integers in [0, {page_count - 1}]")

    @staticmethod
    def _output_dir(source: PDFSource, save_path: str | Path | None) -> Path | None:
        if save_path is None:
            return None

        name = Path(source).stem if not isinstance(source, bytes) else "document_bytes"
        return Path(save_path).expanduser() / name

    def _render_page(
        self,
        page: pymupdf.Page,
        dpi: int,
        output_dir: Path | None,
        overwrite: bool,
    ) -> RasterizedPage:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)

        image = np.frombuffer(pix.samples, dtype=np.uint8)
        image = image.reshape(pix.height, pix.width, 3).copy()
        image_path = None

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"page_{page.number + 1:04d}_{dpi}dpi.png"

            if path.exists() and not overwrite:
                raise FileExistsError(f"Rendered image already exists: {path}")

            pix.save(str(path))
            image_path = str(path.resolve())

        return RasterizedPage(
            page_id=page.number,
            image=image,
            image_path=image_path,
            dpi=dpi,
            width=pix.width,
            height=pix.height,
            pdf_width=float(page.rect.width),
            pdf_height=float(page.rect.height),
        )

    def render_page(
        self,
        source: PDFSource,
        page_id: int = 0,
        dpi: int | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> RasterizedPage:
        render_dpi = self.dpi if dpi is None else dpi
        if render_dpi <= 0:
            raise ValueError("dpi must be positive")

        output_dir = self._output_dir(source, save_path)

        with self._open_pdf(source) as document:
            self._validate_pages([page_id], len(document))
            return self._render_page(document[page_id], render_dpi, output_dir, overwrite)

    def iter_pages(
        self,
        source: PDFSource,
        page_ids: Sequence[int] | None = None,
        dpi: int | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Iterator[RasterizedPage]:
        render_dpi = self.dpi if dpi is None else dpi
        if render_dpi <= 0:
            raise ValueError("dpi must be positive")

        output_dir = self._output_dir(source, save_path)

        with self._open_pdf(source) as document:
            selected = list(range(len(document))) if page_ids is None else list(page_ids)
            self._validate_pages(selected, len(document))

            for page_id in selected:
                yield self._render_page(document[page_id], render_dpi, output_dir, overwrite)

    def render_document(
        self,
        source: PDFSource,
        page_ids: Sequence[int] | None = None,
        dpi: int | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> RasterizedDocument:
        with self._open_pdf(source) as document:
            page_count = len(document)

        pages = list(self.iter_pages(source, page_ids, dpi, save_path, overwrite))
        output_dir = self._output_dir(source, save_path)

        return RasterizedDocument(
            file_path=str(source) if not isinstance(source, bytes) else "<bytes>",
            page_count=page_count,
            pages=pages,
            save_path=str(output_dir.resolve()) if output_dir is not None else None,
        )
