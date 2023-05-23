import unittest

from indicator import bleu_score, rouge_score, bert_score, UnifiedQA_Scorer


class TestIndicator(unittest.TestCase):
    def test_bleu_score(self):
        candidate = ['this', 'is', 'a', 'test']
        references = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
        result = bleu_score(candidate, references)
        self.assertIsInstance(result, dict)

    def test_rouge_score(self):
        candidate = ['this', 'is', 'a', 'test']
        references = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
        result = rouge_score(candidate, references)
        self.assertIsInstance(result, dict)

    def test_bert_score(self):
        candidate = ['this', 'is', 'a', 'test']
        references = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
        result = bert_score(candidate, references)
        self.assertIsInstance(result, dict)


class TestUnifiedQA_Scorer(unittest.TestCase):
    def test_option_score(self):
        scorer = UnifiedQA_Scorer()
        qainput = "What is the capital of France?"
        options = ["Paris", "London", "Berlin"]
        result = scorer.option_score(qainput, options)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(options))


if __name__ == '__main__':
    unittest.main()
