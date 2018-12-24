# -*- coding: utf8 -*-
'''
A Method of Microsoft Concept Graph BLC scoring, 
implement from paper <[CIKM2015] An Inference Approach to Basic Level of Categorization>

@date: 04.12.18
@author: yangbt
'''

import codecs
from scipy.sparse import csr_matrix
from scipy import log, exp


def init_data(inputfile, min_count):
    concept_dict = {}
    entity_dict = {}
    concept_dict_rev = {}
    entity_dict_rev = {}

    concept_row_indices = []
    entity_col_indices = []
    data = []
    total_count = 0
    with codecs.open(inputfile, 'rb', 'utf8') as f:
        for (i, l) in enumerate(f):
            (concept, entity, count) = l.strip().split('\t')
            # skip low count entries
            if int(count) <= min_count:
                continue
            # format raw data
            concept_index = concept_dict.setdefault(concept, len(concept_dict))
            concept_dict_rev[concept_dict[concept]] = concept
            entity_index = entity_dict.setdefault(entity, len(entity_dict))
            entity_dict_rev[entity_dict[entity]] = entity

            concept_row_indices.append(concept_index)
            entity_col_indices.append(entity_index)

            data.append(int(count))
            total_count += int(count)
            if i % 100000 == 0:
                print("Processing : ", i)
    print("Creating matrix ...")
    m = csr_matrix((data, (concept_row_indices, entity_col_indices)), dtype=int)
    return m, total_count, concept_dict_rev, entity_dict_rev


# Global log smoothed probability of entity given concept
def calc_p_e_given_c(m, total_count, eps):
    log_smoothed_p_e_given_c = {}
    for concept_idx in range(m.shape[0]):
        concept_row = m.getrow(concept_idx)
        concept_row_sum = float(concept_row.sum())
        if concept_row_sum > 0:
            for entity_idx in concept_row.nonzero()[1]:
                log_smoothed_p_e_given_c[(concept_idx, entity_idx)] = log(concept_row[0, entity_idx] + eps)\
                                                                      - log(concept_row_sum + eps * total_count)
        if concept_idx % 10000 == 0:
            print("Computed %d/%d p(e|c)" % (concept_idx, m.shape[0]))
    return log_smoothed_p_e_given_c


# Global log probability of concept given entity
def calc_p_c_given_e(m):
    log_p_c_given_e = {}
    m_by_col = m.tocsc()
    for entity_idx in range(m_by_col.shape[1]):
        entity_col = m_by_col.getcol(entity_idx)
        entity_col_sum = entity_col.sum()
        if entity_col_sum > 0:
            for concept_idx in entity_col.nonzero()[0]:
                log_p_c_given_e[(concept_idx, entity_idx)] = log(entity_col[concept_idx, 0])\
                                                             - log(entity_col_sum)
        if entity_idx % 10000 == 0:
            print("Computed %d/%d p(c|e)" % (entity_idx, m_by_col.shape[1]))
    return log_p_c_given_e


# Results writer
def write_results(outputfile, m, log_smoothed_p_e_given_c, log_p_c_given_e, concept_dict_rev, entity_dict_rev):
    with codecs.open(outputfile, 'wb', 'utf8') as f:
        for (i, (concept_idx, entity_idx)) in enumerate(log_smoothed_p_e_given_c):
            if (concept_idx, entity_idx) in log_p_c_given_e:
                rep = log_smoothed_p_e_given_c[(concept_idx, entity_idx)] + log_p_c_given_e[(concept_idx, entity_idx)]
                # write
                f.write("%s\t%s\t%d\t%.6f\t%.6f\t%.6f\n" % (concept_dict_rev[concept_idx],
                                                            entity_dict_rev[entity_idx],
                                                            m[concept_idx, entity_idx],
                                                            exp(log_p_c_given_e[(concept_idx, entity_idx)]),
                                                            exp(log_smoothed_p_e_given_c[(concept_idx, entity_idx)]),
                                                            exp(rep)))
            if i % 100000 == 0:
                print("Wrote %d/%d lines" % (i, len(log_smoothed_p_e_given_c)))

if __name__ == '__main__':

    min_count = 0
    eps = 0.0001

    inputfile = "/home/yangbt/concept_extract_test/data/open_domain/data-concept/data-concept-instance-relations.txt"
    outputfile = "/home/yangbt/concept_extract_test/data/open_domain/data-concept/data-concept-instance-relations-with-blc.txt"

    (m, total_count, concept_dict_rev, entity_dict_rev) = init_data(inputfile, min_count)

    # Smoothed P(e|c) = N(c, e) + esp / (Sum of e_i N(c, e_i) + eps * total_instances)
    # BLC = P(e|c) * P(c|e)
    print("Computing BLC ...")
    log_smoothed_p_e_given_c = calc_p_e_given_c(m, m.shape[1], eps)
    log_p_c_given_e = calc_p_c_given_e(m)

    print("Writing results ...")
    write_results(outputfile, m, log_smoothed_p_e_given_c, log_p_c_given_e, concept_dict_rev, entity_dict_rev)
    print("Done.")
