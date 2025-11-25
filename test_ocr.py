from doctr.io import read_pdf
from doctr.models import ocr_predictor

# Modello LIGHT, inizializzato una sola volta
doctr_ocr = ocr_predictor(
    det_arch="db_mobilenet_v3_large",
    reco_arch="crnn_mobilenet_v3",
    pretrained=True,
)

def extract_pdf_text_with_tables(pdf_path: str) -> str:
    """
    Estrae tutto il testo del PDF usando DocTR in modo più leggero:
    - rendering a risoluzione moderata (scale=1.0)
    - elaborazione pagina per pagina
    - niente logica speciale per le tabelle
    """
    # Renderizza il PDF in immagini (numpy) a bassa/media risoluzione
    pages = read_pdf(pdf_path, scale=1.0)  # prova 0.8 se è ancora lento

    all_pages_text: list[str] = []

    for page_idx, page_img in enumerate(pages):
        print(f"[DOCTR] OCR pagina {page_idx+1}/{len(pages)}")

        # DocTR accetta una lista di pagine (H x W x 3)
        result = doctr_ocr([page_img])

        page_lines: list[str] = []

        # result.pages[0] perché qui passiamo una sola pagina
        for block in result.pages[0].blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                if line_text.strip():
                    page_lines.append(line_text)

        page_text = "\n".join(page_lines)
        all_pages_text.append(page_text)

    return "\n\n".join(all_pages_text)
