# -*- coding: utf8 -*-
'''
Medical Knowledge Graph building, 
implement from paper <[SIGMOD2012] Probase: a probabilistic taxonomy for text understanding>

@date: 12.12.18
@author: yangbt
'''

import pickle
import re
import string

import spacy
from tqdm import tqdm


class Chunk:
    """Class containing chunk and its root."""

    def __init__(self, chunk, chunk_root):
        self.chunk = chunk
        self.chunk_root = chunk_root

    def __eq__(self, other):
        return self.chunk == other.chunk and self.chunk_root == other.chunk_toot

    def __hash__(self):
        return hash((self.chunk, self.chunk_root))

    def __str__(self):
        return "text: " + self.chunk + " root: " + self.chunk_root


class MedProbase:
    """Class to generate Medical knowledge graph named MedProbase """

    def __init__(self):
        # dictionary which stores the count of a super concept
        self.n_super_concept = {}

        # dictionary which stores the count of pairs
        # x, y represents super-concept and sub-concept
        self.n_super_concept_sub_concept = {}

        self.knowledge_base_size = 1

        self.super_concepts_corpus = []
        self.sub_concept_corpus = []

    def syntactic_extraction(self, filename):
        """Extracts sub-concepts and super-concepts from the corpus"""

        patterns1 = []

    def en_syntactic_extration(self, filename):
        """Extracts sub-concepts and super-concepts from the corpus -- English"""

        patterns1 = [r'(.*) such as (.*)',
                     r'(.*) including (.*)',
                     r'(.*) especially (.*)']
        patterns2 = [r'(.*) and other (.*)',
                     r'(.*) or other (.*)']

        nlp = spacy.load('en')

        with open(filename, 'r') as corpus:
            for sentence in tqdm(corpus):
                sentence_parsed = False
                # parsing patterns 1
                for pattern in patterns1:
                    super_concepts = []
                    match = re.match(pattern, sentence)
                    if match:
                        if not sentence_parsed:
                            parsed_sentence = nlp(sentence)
                        for chunk in parsed_sentence.noun_chunks:
                            if chunk.text in match.group(1) and (chunk.root.tag_ == 'NNS' or chunk.root.tag_ == 'NNPS'):
                                super_concepts.append(Chunk(chunk.text.lower(), chunk.root.text.lower()))
                    if super_concepts:
                        self.super_concepts_corpus.append(super_concepts)
                        sub_concepts = []
                        for sub_concept in match.group(2).split(","):
                            sub_concept = sub_concept.strip()
                            sub_concept = sub_concept.lower()
                            sub_concept = sub_concept.translate(str.maketrans("", "", string.punctuation))
                            sub_concepts.append(sub_concept)
                        self.sub_concept_corpus.append(sub_concepts)
                # parsing patterns 2
                for pattern in patterns2:
                    super_concepts = []
                    match = re.match(pattern, sentence)
                    if match:
                        if not sentence_parsed:
                            parsed_sentence = nlp(sentence)
                        for chunk in parsed_sentence.noun_chunks:
                            if chunk.text in match.group(2) and (chunk.root.tag_ == 'NNS' or chunk.root.tag_ == 'NNPS'):
                                super_concepts.append(Chunk(chunk.text.lower(), chunk.root.text.lower()))
                    if super_concepts:
                        self.super_concepts_corpus.append(super_concepts)
                        sub_concepts = []
                        for sub_concept in match.group(1).split(','):
                            sub_concept = sub_concept.strip()
                            sub_concept = sub_concept.lower()
                            sub_concept = sub_concept.translate(str.maketrans("", "", string.punctuation))
                            sub_concepts.append(sub_concept)
                        self.sub_concept_corpus.append(sub_concepts[::-1])

                match = re.match(r'(.*) such (.*) as (.*)', sentence)
                super_concepts = []
                if match:
                    if not sentence_parsed:
                        parsed_sentence = nlp(sentence)
                    for chunk in parsed_sentence.noun_chunks:
                        if chunk.text in match.group(2) and (chunk.root.tag_ == 'NNS' or chunk.root.tag_ == 'NNPS'):
                            super_concepts.append(Chunk(chunk.text.lower(), chunk.root.text.lower()))
                if super_concepts:
                    self.super_concepts_corpus.append(super_concepts)
                    sub_concepts = []
                    for sub_concept in match.group(3).strip(','):
                        sub_concept = sub_concept.rstrip()
                        sub_concept = sub_concept.lower()
                        sub_concept = sub_concept.translate(str.maketrans("", "", string.punctuation))
                        sub_concepts.append(sub_concept)
                    self.sub_concept_corpus.append(sub_concepts)

    def load_concept_corpus(self, super_concepts_corpus, sub_concepts_corpus):
        """Load concepts"""

        self.super_concepts_corpus = super_concepts_corpus
        self.sub_concept_corpus = sub_concepts_corpus

    def p_x(self, super_concept):
        """Return probability of super-concept in knowledgebase"""

        probability = self.n_super_concept.get(super_concept.chunk, 0) / self.knowledge_base_size
        if super_concept.chunk != super_concept.chunk_root:
            probability_root = self.n_super_concept.get(super_concept.chunk_root, 0) / self.knowledge_base_size
            probability += probability_root

        if probability == 0:
            return self.epsilon
        else:
            return probability

    def p_y_x(self, sub_concept, super_concept):
        """Return probability of sub-concept given a super-concept in knowledgebase"""

        probability = self.n_super_concept_sub_concept\
                          .get((super_concept.chunk, sub_concept), 0) / self.n_super_concept.get(super_concept.chunk, 1)
        if super_concept.chunk != super_concept.chunk_root:
            probability_root = self.n_super_concept_sub_concept\
                                   .get((super_concept.chunk_root, sub_concept), 0) / self.n_super_concept.get(super_concept.chunk_root, 1)
            probability += probability_root

        if probability == 0:
            return self.epsilon
        else:
            return probability

    def super_concept_detection(self, super_concepts, sub_concepts):
        """Return most likely super concept"""

        likelihoods = {}
        for super_concept in super_concepts:
            probability_super_concept = self.p_x(super_concept)
            likelihood = probability_super_concept
            for sub_concept in sub_concepts:
                probability_y_x = self.p_y_x(sub_concept, sub_concept)
                likelihood *= probability_y_x
            likelihoods[super_concept] = likelihood

        sorted_likelihoods = sorted(likelihoods.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_likelihoods) == 1:
            return super_concepts[0]
        if sorted_likelihoods[1][1] == 0:
            return sorted_likelihoods[0][0]
        ratio = sorted_likelihoods[0][1] / sorted_likelihoods[1][1]
        if ratio > self.threshold_super_concept:
            return sorted_likelihoods[0][0]
        else:
            return None

    def sub_concept_detection(self, most_likely_super_concept, sub_concepts):
        """Return a list of filtered sub concepts"""

        for k, sub_concept in enumerate(sub_concepts):
            if self.p_y_x(sub_concept, most_likely_super_concept) < self.threshold_k:
                break
        k = max(k, 1)
        return sub_concepts[:k]

    @staticmethod
    def increase_count(dictionary, key):
        """Increases count of key in dictionary"""
        if key in dictionary:
            dictionary[key] += 1
        else:
            dictionary[key] = 1

    def build_knowledge_base(self, epsilon=0.01, threshold_super_concept=1.1, threshold_k=0.001):
        """Takes in list of subconcepts and superconcepts and generates knowledge graph"""

        self.threshold_super_concept = threshold_super_concept
        self.epsilon = epsilon
        self.threshold_k = threshold_k

        iteration = 0
        while True:
            iteration += 1
            n_super_concept_sub_concept_new = {}
            n_super_concept_new = {}
            knowledge_base_size_new = 1
            print("Building iteration: {}".format(iteration))
            for super_concepts, sub_concepts in tqdm(zip(self.super_concepts_corpus, self.sub_concept_corpus)):
                most_likely_super_concept = self.super_concept_detection(super_concepts, sub_concepts)
                if most_likely_super_concept is None:
                    continue
                sub_concepts = self.sub_concept_detection(most_likely_super_concept, sub_concepts)
                for sub_concept in sub_concepts:
                    self.increase_count(n_super_concept_sub_concept_new, (most_likely_super_concept.chunk, sub_concept))
                    self.increase_count(n_super_concept_new, most_likely_super_concept)
                    knowledge_base_size_new += 1
            size_old = len(self.n_super_concept_sub_concept)
            size_new = len(n_super_concept_sub_concept_new)
            if size_new == size_old:
                break
            else:
                self.n_super_concept_sub_concept = n_super_concept_sub_concept_new
                self.n_super_concept = n_super_concept_new
                self.knowledge_base_size = knowledge_base_size_new

    def save_file(self, filename):
        """Saves MedProbase as filename in text format"""
        with open(filename, 'w') as f:
            for key, value in self.n_super_concept_sub_concept.items():
                f.write(key[0] + '\t' + key[1] + '\t' + str(value) + '\n')


if __name__ == '__main__':
    corpus_file = ""
    output_file = ""
    epsilon = 0.01
    threshold_super_concept = 1.1
    threshold_k = 0.001

    med_probase = MedProbase()
    print("Extracting super-concepts and sub-concepts...")
    med_probase.syntactic_extraction(corpus_file)
    print("\nBuilding knowledge graph...")
    med_probase.build_knowledge_base(epsilon, threshold_super_concept, threshold_k)
    print("\nSaving medical probase as text file...")
    med_probase.save_file(output_file)
