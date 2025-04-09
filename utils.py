import config
from typing import List, Optional

def sentence_based_split(
    text: str,
    target_word_count: int = 256,
    overlap_words: int = 32,
    max_tokens: int = 512
) -> List[str]:
    """
    Splits text by sentences with overlap.
    """
    if config.GLOBAL_TOKENIZER is None:
        config.init_global_components()
    tokenizer = config.GLOBAL_TOKENIZER

    doc = config.GLOBAL_NLP(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk_words = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) > 100:
            sub_sentences = [sentence[i : i + 100] for i in range(0, len(words), 100)]
            words = " ".join(sub_sentences).split()

        if len(current_chunk_words) + len(words) > target_word_count:
            chunk_text = " ".join(current_chunk_words)
            if len(tokenizer.encode(chunk_text)) <= max_tokens:
                chunks.append(chunk_text)
            else:
                truncated_words = current_chunk_words.copy()
                while truncated_words and len(tokenizer.encode(" ".join(truncated_words))) > max_tokens:
                    truncated_words.pop()
                if truncated_words:
                    chunks.append(" ".join(truncated_words))

            if len(current_chunk_words) >= overlap_words:
                current_chunk_words = current_chunk_words[-overlap_words:]
            current_chunk_words.extend(words)
        else:
            current_chunk_words.extend(words)

    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        if len(tokenizer.encode(chunk_text)) <= max_tokens:
            chunks.append(chunk_text)
        else:
            truncated_words = current_chunk_words.copy()
            while truncated_words and len(tokenizer.encode(" ".join(truncated_words))) > max_tokens:
                truncated_words.pop()
            if truncated_words:
                chunks.append(" ".join(truncated_words))

    return chunks


def adaptive_sentence_based_split(
    text: str,
    target_word_count: Optional[int] = None,
    overlap_words: Optional[int] = None,
    max_tokens: int = 512
) -> List[str]:
    """
    Adaptive splitting based on token density.
    """
    if config.GLOBAL_TOKENIZER is None:
        config.init_global_components()
    tokenizer = config.GLOBAL_TOKENIZER

    doc = config.GLOBAL_NLP(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    chunks = []
    current_chunk_words = []

    words = text.split()
    total_tokens = len(tokenizer.encode(text))
    avg_tokens_per_word = total_tokens / len(words) if words else 1

    if target_word_count is None:
        target_word_count = int((max_tokens * 0.8) / avg_tokens_per_word)
    if overlap_words is None:
        overlap_words = int(0.2 * target_word_count)

    for sentence in sentences:
        words_list = sentence.split()
        if len(words_list) > target_word_count:
            sub_sentences = [
                words_list[i: i + target_word_count]
                for i in range(0, len(words_list), target_word_count)
            ]
            for sub_words in sub_sentences:
                if len(current_chunk_words) + len(sub_words) > target_word_count:
                    chunk_text = " ".join(current_chunk_words)
                    while current_chunk_words and len(tokenizer.encode(chunk_text)) > max_tokens:
                        current_chunk_words.pop()
                        chunk_text = " ".join(current_chunk_words)
                    if current_chunk_words:
                        chunks.append(chunk_text)
                    current_chunk_words = current_chunk_words[-overlap_words:]
                    current_chunk_words.extend(sub_words)
                else:
                    current_chunk_words.extend(sub_words)
        else:
            if len(current_chunk_words) + len(words_list) > target_word_count:
                chunk_text = " ".join(current_chunk_words)
                while current_chunk_words and len(tokenizer.encode(chunk_text)) > max_tokens:
                    current_chunk_words.pop()
                    chunk_text = " ".join(current_chunk_words)
                if current_chunk_words:
                    chunks.append(chunk_text)
                current_chunk_words = current_chunk_words[-overlap_words:]
                current_chunk_words.extend(words_list)
            else:
                current_chunk_words.extend(words_list)

    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        while current_chunk_words and len(tokenizer.encode(chunk_text)) > max_tokens:
            current_chunk_words.pop()
            chunk_text = " ".join(current_chunk_words)
        if current_chunk_words:
            chunks.append(chunk_text)

    return chunks
