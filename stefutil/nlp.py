import spacy
from tqdm import tqdm

from .prettier import get_logger, ca


__all__ = ['TextPreprocessor']


_logger = get_logger(__name__)


class TextPreprocessor:
    """
    Pre-process documents in to lists of tokens

    By default, the document is broken into words, only non-stop words with alphabets are kept, and words are lemmatized & lowercased
    """
    # tags to remove from the text
    tags_ignore = [
        'ADV',  # adverbs, e.g. extremely, loudly, hard
        'PRON',  # pronouns, e.g. I, you, he
        'CCONJ',  # coordinating conjunctions, e.g. and, or, but
        'PUNCT',  # punctuation
        'PART',  # particle, e.g. about, off, up
        'DET',  # determiner, e.g. a, the, these
        'ADP',  # adposition, e.g. in, to, during
        'SPACE',  # space
        'NUM',  # numeral
        'SYM'  # symbol
    ]
    # from spacy import glossary
    # tag_name = 'ADV'
    # mic(glossary.explain(tag_name))
    # definitions linked in the source code https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    # http://universaldependencies.org/u/pos/

    nlp = None

    def __init__(self, tokenize_scheme: str = 'word', drop_tags: bool = False, verbose: bool = False):
        ca.check_mismatch(display_name='Tokenization Scheme', val=tokenize_scheme, accepted_values=['word', '2-gram', 'chunk'])
        self.tokenize_scheme = tokenize_scheme

        if TextPreprocessor.nlp is None:
            TextPreprocessor.nlp = self.nlp = spacy.load('en_core_web_sm')
        else:
            self.nlp = TextPreprocessor.nlp
        self.nlp.add_pipe("merge_entities")
        self.nlp.add_pipe("merge_noun_chunks")
        self.drop_tags = drop_tags
        self.verbose = verbose

    def keep_token(self, tok) -> bool:
        ret = not tok.is_stop and tok.is_alpha
        if self.drop_tags:
            ret = ret and tok.pos_ not in TextPreprocessor.tags_ignore
        return ret

    def __call__(self, texts: List[str]) -> List[List[str]]:
        avg_tok_len = round(np.mean([len(sentence2tokens(sent)) for sent in texts]), 2)
        ret = []
        it = tqdm(self.nlp.pipe(texts), desc='Preprocessing documents', total=len(texts))
        for doc in it:
            # doc on attributes of a token at https://spacy.io/api/token
            # ignore certain tags & stop words, keep tokens w/ english letters, lemmatize & lowercase
            if self.tokenize_scheme == 'chunk':
                toks = [chunk.text for chunk in doc.noun_chunks]
            else:  # `word`, `2-gram`
                toks = [tok.lemma_.lower() for tok in doc if self.keep_token(tok)]
                if self.tokenize_scheme == '2-gram':
                    toks = [' '.join(toks[i:i + 2]) for i in range(len(toks) - 1)]
            ret.append(toks)
        avg_tok_len_ = round(np.mean([len(toks) for toks in ret]), 2)
        if self.verbose:
            _logger.info(f'Preprocessing finished w/ average token length {pl.i(avg_tok_len)} => {pl.i(avg_tok_len_)}')
        return ret
