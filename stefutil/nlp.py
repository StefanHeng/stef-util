import re
import math
from typing import List, Union, Any

from tqdm import tqdm

from stefutil.container import group_n
from stefutil.prettier import get_logger, s, ca
from stefutil.packaging import _use_dl


__all__ = ['punc_tokenize']


if _use_dl():
    __all__ += ['TextPreprocessor', 'SbertEncoder']


_logger = get_logger(__name__)


# _pattern_space = re.compile(r'\s+')  # match one or more whitespace
_pattern_term = re.compile(r'\w+|[^\s\w]+')  # match one or more alphanumeric or non-whitespace-non-alphanumeric
_pattern_term_split = re.compile(r'(\w+|[^\s\w])')  # adjacent punctuations are separate tokens
_pattern_term_digit = re.compile(r'\w+|[^\s\w]+|\d')  # match one or more alphanumeric, non-whitespace-non-alphanumeric, or digits
_pattern_term_split_digit = re.compile(r'(\w+|[^\s\w]|\d)')  # adjacent punctuations and digits are separate tokens


def punc_tokenize(sentence: str, split_adjacent_puncs: bool = True, split_digits: bool = False) -> List[str]:
    """
    Split sentence into tokens, split on any punctuation or whitespace
        e.g. `SOCCER-JAPAN` => [`SOCCER`, `-`, `JAPAN`]

    :param sentence: sentence to tokenize
    :param split_adjacent_puncs: whether to split adjacent punctuations into separate tokens
    :param split_digits: whether to split digits into separate tokens
    """
    if split_adjacent_puncs:
        pat = _pattern_term_split_digit if split_digits else _pattern_term_split
    else:
        pat = _pattern_term_digit if split_digits else _pattern_term
    return pat.findall(sentence)


if _use_dl():
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
        # sic(glossary.explain(tag_name))
        # definitions linked in the source code https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
        # http://universaldependencies.org/u/pos/

        nlp = None

        def __init__(self, tokenize_scheme: str = 'word', drop_tags: bool = False, verbose: bool = False, prefer_gpu: bool = None):
            ca.assert_options(display_name='Tokenization Scheme', val=tokenize_scheme, options=['word', '2-gram', 'chunk'])
            self.tokenize_scheme = tokenize_scheme

            import spacy  # lazy import to save time
            if prefer_gpu:
                spacy.prefer_gpu()
            if TextPreprocessor.nlp is None:
                TextPreprocessor.nlp = self.nlp = spacy.load('en_core_web_sm')
            else:
                self.nlp = TextPreprocessor.nlp
            # self.nlp.add_pipe("merge_entities")
            # self.nlp.add_pipe("merge_noun_chunks")
            self.drop_tags = drop_tags
            self.verbose = verbose

        def keep_token(self, tok) -> bool:
            ret = not tok.is_stop and tok.is_alpha
            if self.drop_tags:
                ret = ret and tok.pos_ not in TextPreprocessor.tags_ignore
            return ret

        def __call__(self, texts: List[str]) -> List[List[str]]:
            import numpy as np  # lazy import to save time
            assert isinstance(texts, list) and len(texts) > 0
            avg_tok_len = round(np.mean([len(punc_tokenize(sent)) for sent in texts]), 2)
            ret = []
            it = tqdm(self.nlp.pipe(texts), desc='Preprocessing documents', unit=s.i('doc'), total=len(texts))
            for doc in it:
                toks = self.process_single(doc)
                it.set_postfix(tok_len=s.i(len(toks)))
                ret.append(toks)
            avg_tok_len_ = round(np.mean([len(toks) for toks in ret]), 2)
            if self.verbose:
                _logger.info(f'Preprocessing finished w/ average token length {s.i(avg_tok_len)} => {s.i(avg_tok_len_)}')
            return ret

        def process_single(self, text: Union[str, Any]) -> List[str]:
            from spacy.tokens import Doc
            text: Union[str, Doc]
            doc = self.nlp(text) if isinstance(text, str) else text

            # doc on attributes of a token at https://spacy.io/api/token
            # ignore certain tags & stop words, keep tokens w/ english letters, lemmatize & lowercase
            if self.tokenize_scheme == 'chunk':
                toks = [chunk.text for chunk in doc.noun_chunks]
            else:  # `word`, `2-gram`
                toks = [tok.lemma_.lower() for tok in doc if self.keep_token(tok)]
                if self.tokenize_scheme == '2-gram':
                    toks = [' '.join(toks[i:i + 2]) for i in range(len(toks) - 1)]
            return toks


    class SbertEncoder:
        """
        Encode texts w/ SBert, a wrapper around `SentenceTransformer`
        """
        model_name2model = dict()  # class-level model cache

        def __init__(self, model_name: str = 'all-mpnet-base-v2'):
            # per SBert package, the one with the highest quality
            self.model_name = model_name

        @property
        def model(self):
            if self.model_name not in SbertEncoder.model_name2model:
                import torch  # lazy import to save time
                from sentence_transformers import SentenceTransformer

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                SbertEncoder.model_name2model[self.model_name] = SentenceTransformer(self.model_name, device=device)
            return SbertEncoder.model_name2model[self.model_name]

        def __call__(self, texts: List[str], batch_size: int = 32, desc: str = None, **kwargs):
            """
            :param texts: List of texts to encode
            :param batch_size: encode batch size
            :param desc: description for tqdm
            """
            import numpy as np  # lazy import to save time
            n, bsz = len(texts), batch_size
            n_ba = math.ceil(n / bsz)

            lst_vects = np.empty(n_ba, dtype=object)
            it = tqdm(group_n(texts, bsz), total=n_ba, desc=desc, unit='ba')
            it.set_postfix(n=s.i(n), bsz=s.i(bsz))

            enc_args = dict(batch_size=bsz, show_progress_bar=False, )
            enc_args.update(kwargs)
            for i, sents in enumerate(it):
                lst_vects[i] = self.model.encode(sents, **enc_args)
            return np.concatenate(lst_vects, axis=0)


if __name__ == '__main__':
    from stefutil.prettier import sic

    def check_process():
        tp = TextPreprocessor()

        docs = ['hello world', 'japan-soccer.']
        lst_toks = tp(docs)
        sic(lst_toks)

        sic(tp.process_single('hello world'))
    # check_process()

    def check_punc_tokenize():
        # txt = 'Are there any romantic movies from the 90s on Disney+?'
        txt = 'The serotonin 6 (5-HT6) receptor is therapeutically targeted by'
        sic(punc_tokenize(txt))
    check_punc_tokenize()

    def check_enc():
        texts = ['hello world 1', 'hello world 2', 'hello world 3'] * 4
        se = SbertEncoder()
        vects = se(texts, batch_size=2)
        sic(vects.shape)
    # check_enc()
