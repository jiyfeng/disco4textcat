// textclass.h
// Author: Yangfeng Ji
// Date: Oct. 9, 2016
// Time-stamp: <yangfeng 06/22/2017 11:15:34>

#ifndef TEXTCLASS_H
#define TEXTCLASS_H

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/pretrain.h"

#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace dynet;


dynet::Dict d;
// int kSOS;
// int kEOS;

template <class Builder>
struct TextClass {
  LookupParameter p_W; // word embeddings
  // LookupParameter p_Ua; // relation specific attention weight
  Parameter p_Ua; // attention weight
  LookupParameter p_Ut; // relation-specific composition function
  Parameter p_Uc; // classification weight
  Parameter p_bias; // classification bias
  // Builder docbuilder; // builder for the doc structure
  Builder fw_senbuilder; // builder for the forward sent rep
  Builder bw_senbuilder; // builder for the backward sent rep
  unordered_map<int, vector<float>> embeddings; // word embeddings
  bool b_pretrained; // whether use pretrained word embeddings
  unsigned march; // model architecture

public:
  TextClass(Model& model, unsigned input_dim, unsigned hidden_dim, unsigned nlayer,
	    unsigned nclass, unsigned nrela, unsigned vocab_size,
	    Dict& d, const string fembed, const unsigned model_arch) {
    /********************************************************************
     * model: dynet model
     * input_dim: input dimension
     * hidden_dim: hidden dimension
     * nlayer: number of layers of LSTM builders
     * nclass: number of text categories
     * nrela: number of discourse relations
     * vocab_size: vocab size
     * d: dynet dictionary
     * fembed: word embedding file
     * model_arch: model arch index (0: full model; 1: without comp mat; 
     *                               2: single edu)
     ********************************************************************/
    p_W = model.add_lookup_parameters(vocab_size, {input_dim});
    // p_Ua = model.add_lookup_parameters(nrela, {hidden_dim*4});
    p_Ua = model.add_parameters({hidden_dim*2, hidden_dim*2});
    p_Ut = model.add_lookup_parameters(nrela, {hidden_dim*2, hidden_dim*2});
    p_Uc = model.add_parameters({nclass, hidden_dim*2});
    p_bias = model.add_parameters({nclass}, 0.0);
    // docbuilder = Builder(nlayer, hidden_dim*2, hidden_dim*2, model);
    fw_senbuilder = Builder(nlayer, input_dim, hidden_dim, model);
    bw_senbuilder = Builder(nlayer, input_dim, hidden_dim, model);
    if (fembed.size() > 0){
      read_pretrained_embeddings(fembed, d, embeddings);
      b_pretrained = true;
      cerr << "Load " << embeddings.size() << " word embeddings with dim:" << embeddings[0].size() << endl;
      assert(embeddings[0].size() == input_dim);
    } else {
      b_pretrained = false;
    }
    march = model_arch;
    if ((march > 4) or (march < 0)){
      cerr << "Unrecognized model architecture index: " << march << endl;
      exit(1);
    }
  }

  // main function to build a CG
  Expression build_model(Doc&, ComputationGraph&, float, bool, Record&);

private:
  // build sentence reps
  vector<Expression> build_edus(const Doc&, ComputationGraph&, float, bool);
  
};

template <class Builder>
Expression TextClass<Builder>::build_model(Doc& doc,
					   ComputationGraph& cg,
					   float dropout_rate,
					   bool b_test,
					   Record& record){
  // add builder-based expression to the graph
  bool b_dropout = ((dropout_rate > 0) and (!b_test));
  Expression Ua = parameter(cg, p_Ua);
  // cerr << "whether dropout: " << b_dropout << endl;
  // network dropout
  // if (b_dropout) docbuilder.set_dropout(dropout_rate);
  // docbuilder.new_graph(cg); 
  // docbuilder.start_new_sequence();
  // get all EDU representations
  // cerr << "build sentence representations ..." << endl;
  vector<Expression> edus = build_edus(doc, cg, dropout_rate, b_dropout);
  // cerr << "number of EDUs: " << doc.edus.size() << endl;
  // cerr << "edus.size(): " << edus.size() << endl;
  // build representation based tree structure
  // cerr << "build doc representation ..." << endl;
  if (march <= 1){
    for (auto& pidx : doc.order){
      // int pidx = it->first; // parent node
      vector<int> cnodes = doc.tree[pidx]; // a list of children nodes
      if (cnodes.empty()) continue;
      // cerr << "children nodes: ";
      // print_int_vector(cnodes);
      Expression comprep = edus[pidx]; // init comp node
      for (auto cidx : cnodes){
	// get relation index
	int ridx = doc.relas[cidx];
	// if (ridx >= 1) ridx = 1; // shrink the relation size for debug
	// cerr << "relation index = " << ridx << endl;
	// compute attention weight
	// cerr << "compute attention weight" << endl;
	// Expression alpha = logistic(transpose(lookup(cg, p_Ua, ridx)) * concatenate({edus[pidx], edus[cidx]}));
	// bi-linear form attention weights
	Expression alpha = logistic(transpose(edus[pidx]) * (Ua * edus[cidx]));
	// keep record
	pair<unsigned, float> rec;
	rec.first = cidx;
	rec.second = as_scalar(cg.incremental_forward(alpha));
	record.push_back(rec);
	// cerr << "weighting" << endl;
	Expression weighted_rep;
	if (march == 0){
	  // with composition matrix (more parameters)
	  weighted_rep = lookup(cg, p_Ut, ridx) * edus[cidx];
	  // weighted_rep = lookup(cg, p_Ut, (unsigned)0) * edus[cidx];
	  weighted_rep = weighted_rep * alpha;
	} else {
	  // no composition function
	  weighted_rep = edus[cidx] * alpha;
	}
	// cerr << "adding" << endl;
      comprep = comprep + weighted_rep;
      }
      // take compnode as input to the docbuilder
      // and update the corresponding representation
      // edus[pidx] = docbuilder.add_input(comprep);
      edus[pidx] = tanh(comprep);
    }
  } else if (march == 3){
    // bag-of-EDU model
    for (unsigned eidx = 0; eidx < edus.size(); eidx ++){
      if (eidx != doc.root)
	edus[doc.root] = edus[doc.root] + edus[eidx];
    }
    // take average
    edus[doc.root] = (edus[doc.root] / edus.size());
  } else if (march == 4){ // standard attention
    // cout << "Try standard attention weights ..." << endl;
    for (auto& pidx : doc.order){
      // get parent EDU rep
      Expression comprep = edus[pidx];
      // get all children nodes
      vector<int> cnodes = doc.tree[pidx];
      if (cnodes.size() > 0){
	// if it's not empty
	vector<Expression> cnodes_exps;
	for (auto cidx : cnodes){
	  cnodes_exps.push_back(edus[cidx]);
	}
	// attention weights
	Expression alpha = softmax(transpose(concatenate_cols(cnodes_exps)) * (Ua * edus[pidx]));
	// composition
	for (unsigned iidx = 0; iidx < cnodes.size(); iidx++){
	  Expression weighted_rep = cnodes_exps[iidx] * pick(alpha, iidx);
	  comprep = comprep + weighted_rep;
	}
      }
      edus[pidx] = tanh(comprep);
    }
  } else {
    // what?
    abort();
  }
  // get root node
  Expression root = edus[doc.root];
  // output dropout
  if (b_dropout) root = dropout(root, dropout_rate);
  // get classification parameters
  Expression Uc = parameter(cg, p_Uc);
  Expression bias = parameter(cg, p_bias);
  // compuate the log-prob
  Expression logit = (Uc * root) + bias;
  if (b_test){
    Expression prob = softmax(logit);
    return prob;
  } else {
    Expression p_err = pickneglogsoftmax(logit, doc.label);
    return p_err;
  }
}

/*******************************************************
 * build reps for all EDUs
 *******************************************************/
template <class Builder>
vector<Expression> TextClass<Builder>::build_edus(const Doc& doc,
						  ComputationGraph& cg,
						  float dropout_rate,
						  bool b_dropout){
  if (b_dropout){
    fw_senbuilder.set_dropout(dropout_rate);
    bw_senbuilder.set_dropout(dropout_rate);
  }
  fw_senbuilder.new_graph(cg);
  bw_senbuilder.new_graph(cg);
  vector<Edu> edus = doc.edus;
  unsigned n_edus = edus.size();
  vector<Expression> sent_reps;
  for (unsigned idx = 0; idx < n_edus; idx++){
    // for each sentence
    fw_senbuilder.start_new_sequence();
    bw_senbuilder.start_new_sequence();
    // 
    Edu edu = edus[idx];
    unsigned n_token = edu.size();
    // cerr << "n_token = " << n_token << endl;
    for (int t = 0; t < n_token; t++){
      Expression w_t, h_t;
      if (b_pretrained){
	// cerr << "get from pretrained embeddings " << endl;
	vector<float> v = embeddings[edu[t]];
	// constant input, no update
	w_t = input(cg, {(unsigned)v.size()}, v);
      } else {
	w_t = lookup(cg, p_W, edu[t]);
      }
      // input dropout
      if (b_dropout) w_t = dropout(w_t, dropout_rate);
      h_t = fw_senbuilder.add_input(w_t);
    }
    for (int t = n_token - 1; t > -1; t--){
      Expression w_t, h_t;
      if (b_pretrained){
	vector<float> v = embeddings[edu[t]];
	w_t = input(cg, {(unsigned)v.size()}, v);
      } else {
	w_t = lookup(cg, p_W, edu[t]);
      }
      // input dropout
      if (b_dropout) w_t = dropout(w_t, dropout_rate);
      h_t = bw_senbuilder.add_input(w_t);
    }
    // take the last hidden state as the sent rep
    Expression senrep = concatenate({fw_senbuilder.back(), bw_senbuilder.back()});
    sent_reps.push_back(senrep);
  }
  return sent_reps;
}

#endif
