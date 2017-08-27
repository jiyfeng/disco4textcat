// util.cc
// Author: Yangfeng Ji
// Date: Oct. 14, 2016
// Time-stamp: <yangfeng 12/28/2016 18:52:16>

#include <algorithm>
#include <queue>

#include "util.h"

using namespace std;

#include <boost/algorithm/string.hpp>

Corpus read_corpus(char* filename, dynet::Dict* dptr,
		   bool b_update){
  cerr << "Reading data from " << filename << endl;
  Corpus corpus;
  Doc doc;
  Edu edu;
  string line;
  ifstream in(filename);
  getline(in, line); // get rid of the title line
  // cerr << line << endl;
  while (getline(in, line)){
    if (line.empty()) continue; // just in case
    if (line[0] != '='){
      // within document
      vector<string> items;
      boost::split(items, line, boost::is_any_of("\t"));
      int eidx = std::stoi(items[0]);
      int pidx = std::stoi(items[1]);
      int ridx = std::stoi(items[2]);
      edu = read_edu(items[3], dptr, b_update);
      doc.edus.push_back(edu); // store the edu
      doc.tree[pidx].push_back(eidx); // store the pnode index
      doc.relas[eidx] = ridx; // relation index
      if (pidx == -1) doc.root = eidx; // root node
    } else {
      // end of document
      vector<string> items;
      boost::split(items, line, boost::is_any_of("\t"));
      doc.filename = items[1]; // get filename
      doc.label = std::stoul(items[2]); // get label
      if (doc.edus.size() > 0){
	// before save this doc, get the topological order
	doc.order = topological_sorting(doc);
	// cerr << "filename = " << doc.filename << endl;
	// print_int_vector(doc.order);
	// save this doc
	corpus.push_back(doc);
      } else {
	cerr << "Empty doc: " << doc.filename << endl;
      }
      doc = Doc(); // reset this variable
    }
  }
  if (doc.edus.size() > 0){
    doc.order = topological_sorting(doc);
    cerr << "filename = " << doc.filename << endl;
    print_int_vector(doc.order);
    corpus.push_back(doc);
  }
  cerr << "Read " << corpus.size() << " docs with the vocab has " << dptr->size() << " types" << endl;
  return(corpus);
}


Edu read_edu(const string& line, dynet::Dict* dptr, bool b_update){
  vector<string> tokens;
  boost::split(tokens, line, boost::is_any_of(" "));
  Edu edu;
  // edu.push_back(dptr->convert("<s>"));
  for (auto& tok : tokens){
    if (tok.empty()) continue; // just in case
    if (b_update or dptr->contains(tok)){
      edu.push_back(dptr->convert(tok));
    } else {
      edu.push_back(dptr->convert("UNK"));
    }
  }
  // edu.push_back(dptr->convert("</s>"));
  if (edu.size() == 0){
    // just in case, there is a wired empty sentence
    edu.push_back(dptr->convert("UNK"));
  }
  return edu;
}


vector<int> topological_sorting(Doc& doc){
  vector<int> pnode_list;
  queue<int> q;
  q.push(doc.tree[-1][0]); // add the root node
  // cerr << q.front() << " " << doc.tree[-1][0] << endl;
  // exit(1);
  while(!q.empty()){
    int pidx = q.front();
    // cerr << "pidx = " << pidx << endl;
    q.pop();
    pnode_list.push_back(pidx);
    for (auto& v : doc.tree[pidx]){
      q.push(v);
    }
  }
  // pnode_list.reverse();
  reverse(pnode_list.begin(), pnode_list.end());
  return pnode_list;
}


void print_int_vector(const vector<int>& vec){
  for (auto& val : vec){
    cerr << val << " ";
  }
  cerr << endl;
}


void print_float_vector(const vector<float>& vec){
  for (auto& val : vec){
    cerr << val << " ";
  }
  cerr << endl;
}

// *******************************************************
// save dict from a archive file
// *******************************************************
int save_dict(string fname, dynet::Dict& d){
  // fname += ".dict";
  ofstream out(fname);
  boost::archive::text_oarchive odict(out);
  odict << d; out.close();
  return 0;
}

// *******************************************************
// load dict from a archive file
// *******************************************************
int load_dict(string fname, dynet::Dict& d){
  // fname += ".dict";
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> d; in.close();
  return 0;
}

// *******************************************************
// load model from a archive file
// *******************************************************
int load_model(string fname, Model& model){
  ifstream in(fname);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  return 0;
}

// *******************************************************
// save model from a archive file
// *******************************************************
int save_model(string fname, Model& model){
  ofstream out(fname);
  boost::archive::text_oarchive oa(out);
  oa << model; 
  out.close();
  return 0;
}
