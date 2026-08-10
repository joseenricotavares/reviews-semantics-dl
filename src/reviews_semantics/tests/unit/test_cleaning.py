from reviews_semantics.data.cleaning import clean_text


def test_lowercases_and_strips_punctuation():
    assert clean_text("Ótimo Produto!!! Chegou rápido.") == "ótimo produto chegou rápido"


def test_collapses_whitespace():
    assert clean_text("muito   bom\n\nrecomendo") == "muito bom recomendo"


def test_keeps_portuguese_accented_letters():
    # Matches the accent set the source notebook's regex actually allows
    # (á é í ó ú â ê ô ã õ ç) - notably NOT grave accents like "à".
    assert clean_text("não é   ótimo") == "não é ótimo"


def test_strips_digits_and_symbols():
    assert clean_text("nota 10/10 - excelente!! :)") == "nota excelente"


def test_empty_input_yields_empty_string():
    assert clean_text("") == ""


def test_is_idempotent():
    # data/*.csv is stored pre-cleaned, production input is genuinely raw -
    # both paths call clean_text and must converge to the same tokens.
    once = clean_text("Produto Ótimo!!! chegou rápido.")
    twice = clean_text(once)
    assert once == twice
