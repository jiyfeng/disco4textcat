// util.h
// Author: Yangfeng Ji
// Date: Oct. 9, 2016
// Time-stamp: <yangfeng 02/04/2017 10:38:38>

#ifndef UTIL_H
#define UTIL_H

#include "dynet/dict.h"
#include "dynet/model.h"

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <utility>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace dynet;

// a list of word indices
typedef vector<pair<unsigned, float>> Record;
typedef vector<int> Edu;

struct Doc{
  vector<Edu> edus; // a collection of EDUs
  vector<int> order; // topological order of pnodes
  map<int, vector<int>> tree; // pnode : {children_nodes}
  map<int, int> relas;
  int root; // root node
  unsigned label; // document label
  string filename;
};

typedef vector<Doc> Corpus;

Edu read_edu(const string& line, dynet::Dict* dptr, bool b_update);

Corpus read_corpus(char* filename, dynet::Dict* dptr, bool b_update);

vector<int> topological_sorting(Doc& doc);

void print_int_vector(const vector<int>&);

void print_float_vector(const vector<float>&);

int save_dict(string, dynet::Dict&);

int load_dict(string, dynet::Dict&);

int load_model(string fname, Model& model);

int save_model(string fname, Model& model);

#endif
