// Name: main.cc
// Author: Yangfeng Ji
// Date: Oct. 9, 2016
// Time-stamp: <yangfeng 03/16/2017 16:38:32>

#include "dynet/globals.h"
#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/model.h"

#include "textclass.h"

#include <boost/program_options.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/format.hpp>

namespace po = boost::program_options;

#define _NO_DEBUG_MODE_ 1

// For logging
#if _NO_DEBUG_MODE_
#define ELPP_NO_DEFAULT_LOG_FILE
#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP
#endif

int main(int argc, char** argv) {
  dynet::initialize(argc, argv);

  // argument parsing
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce this help information")
    ("task", po::value<string>(), "task")
    ("trnfile", po::value<string>()->default_value(string("")), "training file")
    ("devfile", po::value<string>()->default_value(string("")), "dev file")
    ("tstfile", po::value<string>()->default_value(string("")), "test file")
    ("dctfile", po::value<string>()->default_value(string("")), "dict file")
    ("modfile", po::value<string>()->default_value(string("")), "model file")
    ("arch", po::value<unsigned>()->default_value((unsigned)0), "model architecture")
    ("nclass", po::value<unsigned>()->default_value((unsigned)15), "number of doc classes")
    ("ndisrela", po::value<unsigned>()->default_value((unsigned)36), "number of discourse relations")
    ("inputdim", po::value<unsigned>()->default_value((unsigned)16), "input dimension")
    ("hiddendim", po::value<unsigned>()->default_value((unsigned)16), "hidden dimension")
    ("nlayer", po::value<unsigned>()->default_value((unsigned)1), "number of hidden layers")
    ("trainer", po::value<unsigned>()->default_value((unsigned)0), "training method")
    ("lr", po::value<float>()->default_value((float)0.1), "learning rate")
    ("droprate", po::value<float>()->default_value((float)0), "dropout rate")
    ("niter", po::value<unsigned>()->default_value((unsigned)1), "number of passes on the training set")
    ("evalfreq", po::value<unsigned>()->default_value((unsigned)1), "evaluation frequency on dev data")
    ("emfile", po::value<string>()->default_value(string("")), "word embedding file")
    ("evaltrn", po::value<bool>()->default_value((bool)false), "evaluation on training data")
    ("path", po::value<string>()->default_value(string("tmp")), "path to save files")
    ("verbose", po::value<bool>()->default_value((bool)false), "print training information");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {cerr << desc << endl; return 1;}
  if (!vm.count("task")) {
    cerr << endl << "Please specify the task, either 'train' or 'test'" << endl;
    return 2;
  }

  // get the arguments
  string task = vm["task"].as<string>();
  string ftrn = vm["trnfile"].as<string>();
  string fdev = vm["devfile"].as<string>();
  string ftst = vm["tstfile"].as<string>();
  string fdct = vm["dctfile"].as<string>();
  string fmod = vm["modfile"].as<string>();
  unsigned arch = vm["arch"].as<unsigned>();
  unsigned nclass = vm["nclass"].as<unsigned>();
  unsigned ndisrela = vm["ndisrela"].as<unsigned>();
  unsigned inputdim = vm["inputdim"].as<unsigned>();
  unsigned hiddendim = vm["hiddendim"].as<unsigned>();
  unsigned trainer = vm["trainer"].as<unsigned>();
  float lr = vm["lr"].as<float>();
  unsigned nlayer = vm["nlayer"].as<unsigned>();
  unsigned niter = vm["niter"].as<unsigned>();
  float droprate = vm["droprate"].as<float>();
  unsigned evalfreq = vm["evalfreq"].as<unsigned>();
  string fembed = vm["emfile"].as<string>();
  bool b_evaltrn = vm["evaltrn"].as<bool>();
  string path = vm["path"].as<string>();
  bool b_verbose = vm["verbose"].as<bool>();

  // get file name
  ostringstream os;
  os << "record" << "-pid" << getpid();
  const string fprefix = path + "/" + os.str();
  const string flog = fprefix + ".log";

  // check file system
  boost::filesystem::path dir(path);
  if(!(boost::filesystem::exists(dir))){
    cerr<< path << " doesn't exist"<<std::endl;
    if (boost::filesystem::create_directory(dir))
      cerr << "Successfully created folder: " << path << endl;
  }

#if _NO_DEBUG_MODE_
  // initialize logging function
  START_EASYLOGGINGPP(argc, argv);
  el::Configurations defaultConf;
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Format, 
  		  "%datetime{%b-%d-%h:%m:%s} %level %msg");
  defaultConf.set(el::Level::Info, 
  		  el::ConfigurationType::Filename, flog.c_str());
  el::Loggers::reconfigureLogger("default", defaultConf);
  
  LOG(INFO) << "[TextClass] training file: " << ftrn;
  LOG(INFO) << "[TextClass] dev file: " << fdev;
  LOG(INFO) << "[TextClass] test file: " << ftst;
  LOG(INFO) << "[TextClass] model file: " << fmod;
  LOG(INFO) << "[TextClass] model architecture: " << arch;
  LOG(INFO) << "[TextClass] number of doc classes: " << nclass;
  LOG(INFO) << "[TextClass] number of discourse relations: " << ndisrela;
  LOG(INFO) << "[TextClass] input dimension: " << inputdim;
  LOG(INFO) << "[TextClass] hidden dimension: " << hiddendim;
  LOG(INFO) << "[TextClass] number of hidden layers: " << nlayer;
  LOG(INFO) << "[TextClass] training method: " << trainer;
  LOG(INFO) << "[TextClass] learning rate: " << lr;
  LOG(INFO) << "[TextClass] number of iterations: " << niter;
  LOG(INFO) << "[TextClass] dropout rate (0: no dropout): " << droprate;
  LOG(INFO) << "[TextClass] evaluation frequency on dev data: " << evalfreq;
  LOG(INFO) << "[TextClass] word embedding file: " << fembed;
  LOG(INFO) << "[TextClass] evaluation on training data: " << b_evaltrn;
  LOG(INFO) << "[TextClass] output path: " << path;
  LOG(INFO) << "[TextClass] verbose: " << b_verbose;
#endif

  // check arguments
  if ((task == "train") and ((ftrn.size() == 0) or (fdev.size() == 0))){
    cerr << "Please specify training and dev files" << endl;
    return 3;
  } else if ((task == "test") and ((ftst.size() == 0) or (fdct.size() == 0) or (fmod.size() == 0))){
    cerr << "Please specify dev, dict and model files" << endl;
    return 4;
  }

  Corpus trncorpus, devcorpus, tstcorpus;
  unsigned vocab_size;
  if (task == "train"){
    // kSOS = d.convert("<s>");
    // kEOS = d.convert("</s>");
    trncorpus = read_corpus((char*)ftrn.c_str(), &d, true);
    d.freeze(); // no new word types allowed
    vocab_size = d.size();
    // cout << "vocab size: " << vocab_size << endl;
#if _NO_DEBUG_MODE_
    LOG(INFO) << "[TextClass] vocab size: " << vocab_size;
#endif
    // save dict
    save_dict(fprefix+".dict", d);
    // read dev corpus
    devcorpus = read_corpus((char*)fdev.c_str(), &d, false);
  } else if (task == "test"){
    // load dict
    load_dict(fdct, d);
    d.freeze();
    vocab_size = d.size();
    // cout << "vocab size " << vocab_size << endl;
#if _NO_DEBUG_MODE_
    LOG(INFO) << "[TextClass] vocab size " << vocab_size;
#endif 
    // load test corpus
    tstcorpus = read_corpus((char*)ftst.c_str(), &d, false);
  } else {
#if _NO_DEBUG_MODE_
    LOG(INFO) << "Unrecognized task label " << task;
#else
    cerr << "Unrecognized task label " << task << endl;
#endif
    return 5;
  }

  // training method
  Model model;
  Trainer* sgd = nullptr;
  if (task == "train"){
    if (trainer == 0){
      sgd = new SimpleSGDTrainer(model, lr);
      // sgd->eta_decay = 0.08;
    } else if (trainer == 1){
      sgd = new AdagradTrainer(model, lr);
    } else if (trainer == 2){
      sgd = new AdamTrainer(model, lr);
    } else {
#if _NO_DEBUG_MODE_
      LOG(INFO) << "Unrecognized trainer " << trainer;
#else
      cerr << "Unrecognized trainer " << trainer << endl;
#endif
      exit(1);
    }
  }
  if (fmod.size() > 0){
    // load pretrained model
    load_model(fmod, model);
  }
  TextClass<LSTMBuilder> tc(model, inputdim, hiddendim, nlayer,
			    nclass, ndisrela, vocab_size,
			    d, fembed, arch);

  // start do sth
  if (task == "train"){
    unsigned reportfreq = 50;
    float best_dev_acc = 0.0;
    vector<unsigned> order(trncorpus.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned si = trncorpus.size();
    niter = (unsigned)(niter*trncorpus.size()/reportfreq);
    while(report < niter) {
      // cout << "Whole training procedure finished: " << boost::format("%1.4f") % (float)report/niter << endl;
      if (b_verbose){
	Timer iteration("completed in");
#if _NO_DEBUG_MODE_
	LOG(INFO) << "Whole training procedure finished: " << boost::format("%1.4f") % ((float)report/niter);
#else
	cout << "Whole training procedure finished: " << boost::format("%1.4f") % ((float)report/niter);
#endif
      }
      double loss = 0;
      unsigned ni = 0;
      for (unsigned i = 0; i < reportfreq; ++i) {
	if (si == trncorpus.size()) {
	  si = 0;
	  if (first) { first = false;} else { sgd->update_epoch();}
	  if (b_verbose) {
#if _NO_DEBUG_MODE_
	    LOG(INFO) << "*** SHUFFLE ***";
#else
	    cout << "*** SHUFFLE ***" << endl;
#endif
	  }
	  shuffle(order.begin(), order.end(), *rndeng);
	}
	
	// build graph for this instance
	ComputationGraph cg;
	auto& doc = trncorpus[order[si]];
	si ++; ni ++;
	Record record;
	Expression loss_expr = tc.build_model(doc, cg, droprate, false, record);
	loss += as_scalar(cg.forward(loss_expr));
	cg.backward(loss_expr);
	sgd->update();
      }
      if (b_verbose){
	sgd->status();
	// cout << "E = " << boost::format("%1.4f") % (loss / ni) << " ";
#if _NO_DEBUG_MODE_	
	LOG(INFO) << "E = " << boost::format("%1.4f") % (loss / ni) << " ";
#else
	cout << "E = " << boost::format("%1.4f") % (loss / ni) << " " << endl;
#endif	
      }
      report++;

      float dev_acc = 0.0;
      if (report % evalfreq == 0) {
	// evaluate on training set
	// if (b_verbose) cerr << endl;
	if (b_evaltrn){
	  float trncorrect = 0;
	  for (auto& doc : trncorpus) {
	    ComputationGraph cg;
	    Record record;
	    Expression loss_expr = tc.build_model(doc, cg, 0.0, true, record);
	    vector<float> prob = as_vector(cg.forward(loss_expr));
	    unsigned plabel = distance(prob.begin(), max_element(prob.begin(), prob.end()));
	    if (plabel == doc.label) trncorrect += 1;
	  }
	  // cout << "Trn accuracy = " << boost::format("%1.4f") % (trncorrect/trncorpus.size()) << endl;
	  if (b_verbose){
#if _NO_DEBUG_MODE_	    
	    LOG(INFO) << "Trn accuracy = " << boost::format("%1.4f") % (trncorrect/trncorpus.size());
#else
	    cout << "Trn accuracy = " << boost::format("%1.4f") % (trncorrect/trncorpus.size()) << endl;
#endif
	  }
	}
	// evaluate on dev set
	float devcorrect = 0;
	ofstream devwfile;
	if (b_verbose) devwfile.open(fprefix + ".devw");
	for (auto& doc : devcorpus) {
	  ComputationGraph cg;
	  Record record;
	  Expression loss_expr = tc.build_model(doc, cg, 0.0, true, record);
	  vector<float> prob = as_vector(cg.forward(loss_expr));
	  unsigned plabel = distance(prob.begin(), max_element(prob.begin(), prob.end()));
	  if (plabel == doc.label) devcorrect += 1;
	  // write dev weight file
	  if (b_verbose){
	    devwfile << "file name = " << doc.filename << endl;
	    devwfile << "label = " << doc.label << "; plabel = " << plabel << endl;
	    for (auto& p : record){
	      devwfile << "(" << p.first << " : " << p.second << ") ";
	    }
	    devwfile << endl;
	  }
	}
	dev_acc = devcorrect/devcorpus.size();
	if (b_verbose){
#if _NO_DEBUG_MODE_
	  LOG(INFO) << "Dev accuracy = " << boost::format("%1.4f") % dev_acc
		    << " ( " << boost::format("%1.4f") % best_dev_acc << " )";
#else
	  cout << "Dev accuracy = " << boost::format("%1.4f") % dev_acc
	       << " ( " << boost::format("%1.4f") % best_dev_acc << " )" << endl;
#endif
	}
	if (dev_acc > best_dev_acc) {
	  // cout << " Save model to: " << fprefix << endl;
	  if (b_verbose){
#if _NO_DEBUG_MODE_
	    LOG(INFO) << "Save model to: " << fprefix;
#else
	    cout << "Save model to: " << fprefix << endl;
#endif
	  }
	  best_dev_acc = dev_acc;
	  save_model(fprefix+".model", model);
	}
	if (b_verbose) devwfile.close();
      }
    }
    // cerr << "Result for SMAC: SUCCESS, 0, 0, " << best_dev_acc << ", 0" << endl;
    // cout << "Final Dev Accuracy : " << boost::format("%1.4f") % best_dev_acc << endl;
#if _NO_DEBUG_MODE_
    LOG(INFO) << "Final Dev Accuracy : " << boost::format("%1.4f") % best_dev_acc;
#else
    cout << "Final Dev Accuracy : " << boost::format("%1.4f") % best_dev_acc << endl;
#endif
    delete sgd;
  } else if (task == "test"){
    int counter = 0;
    float tstcorrect = 0;
    for (auto& doc : tstcorpus){
      counter += 1;
      ComputationGraph cg;
      Record record;
      Expression loss_expr = tc.build_model(doc, cg, droprate, true, record);
      vector<float> prob = as_vector(cg.forward(loss_expr));
      unsigned plabel = distance(prob.begin(), max_element(prob.begin(), prob.end()));
      if (plabel == doc.label) tstcorrect += 1;
      if (b_verbose and (counter % 1000 == 0))
	cout << "Evaluation finished: " << boost::format("%1.2f") % ((float)counter/tstcorpus.size()) << endl;
    }
    float tst_acc = tstcorrect/tstcorpus.size();
    // cout << "Final Test Accuracy : " << boost::format("%1.4f") % tst_acc << endl;
#if _NO_DEBUG_MODE_
    LOG(INFO) << "Final Test Accuracy : " << boost::format("%1.4f") % tst_acc;
#else
    cout << "Final Test Accuracy : " << boost::format("%1.4f") % tst_acc << endl;
#endif
  }
  
} // end of main

