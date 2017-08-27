from rst_reader import RSTReader
from os import listdir
from os.path import join, basename
from collections import defaultdict
from operator import itemgetter
from string import punctuation
import io, gzip


class Doc(object):
    def __init__(self, fname, edus, relas, pnodes, label=None):
        self.fname = fname
        self.edus = edus
        self.relas = relas
        self.pnodes = pnodes
        self.label = label

def parse_fname(fname):
    items = (fname.split(".")[0]).split("-")
    if len(items) != 2:
        raise ValueError("Unexpected length of items: {}".format(items))
    setlabel, fidx = items[0], int(items[1])
    return setlabel, fidx

def load_labels(fname):
    with gzip.open(fname, 'r') as fin:
        labels = fin.read().strip().split("\n")
        print "Load {} labels from file: {}".format(len(labels), fname)
    return labels

def check_token(tok):
    # punctuation only
    is_punc = True
    for c in tok:
        is_punc = is_punc and (c in punctuation)
    # is number
    tok = tok.replace(",","")
    is_number = True
    try:
        float(tok)
    except ValueError:
        is_number = False
    # refine token
    tok = tok.replace("-", "")
    return tok, is_punc, is_number

def get_allbracketsfiles(rpath, suffix=".brackets"):
    bracketsfiles = [join(rpath, fname) for fname in listdir(rpath) if fname.endswith(suffix)]
    print "Read {} files".format(len(bracketsfiles))
    return bracketsfiles


def get_docdict(bracketsfiles, trn_labels, tst_labels, suffix=".brackets"):
    counter = 0
    trn_docdict, tst_docdict = {}, {}
    for fbrackets in bracketsfiles:
        # print "Read file: {}".format(fbrackets)
        fmerge = fbrackets.replace(suffix, ".merge")
        rstreader = RSTReader(fmerge, fbrackets)
        try:
            rstreader.read()
        except SyntaxError:
            # print "Ignore file: ", fmerge
            counter += 1
            continue
        fname = basename(fmerge).replace(".merge","")
        setlabel, fidx = parse_fname(fname)
        if setlabel == "train":
            doc = Doc(fname, rstreader.segtexts, rstreader.textrelas, rstreader.pnodes, int(trn_labels[fidx])-1)
            trn_docdict[fname] = doc
        elif setlabel == "test":
            doc = Doc(fname, rstreader.segtexts, rstreader.textrelas, rstreader.pnodes, int(tst_labels[fidx])-1)
            tst_docdict[fname] = doc
    print "Ignore {} files in total".format(counter)
    return trn_docdict, tst_docdict

def get_vocab(docs, thresh=10000):
    counts = defaultdict(int)
    rela_vocab = {'root':0}
    for (fname, doc) in docs.iteritems():
        for (eidx, edu) in doc.edus.iteritems():
            tokens = edu.strip().split()
            for tok in tokens:
                tok, is_punc, is_number = check_token(tok)
                if is_punc:
                    continue
                if is_number:
                    tok = "NUMBER"
                counts[tok] += 1
        for (eidx, rela) in doc.relas.iteritems():
            try:
                rela_vocab[rela]
            except KeyError:
                rela_vocab[rela] = len(rela_vocab)
    print "Size of the raw vocab: ", len(counts)
    # rank with 
    sorted_counts = sorted(counts.items(), key=itemgetter(1), reverse=True)
    word_vocab, N = {}, 0
    for item in sorted_counts:
        word_vocab[item[0]] = N
        N += 1
        if N >= thresh:
            break
    print "After filtering out low-frequency words, vocab size = ", len(word_vocab)
    return word_vocab, rela_vocab


def refine_with_vocab(edu, vocab):
    t_count, u_count = 0.0, 0.0
    tokens = edu.strip().split()
    new_tokens = []
    for tok in tokens:
        tok, is_punc, is_number = check_token(tok)
        if is_punc:
            continue
        t_count += 1.0
        if is_number:
            tok = "NUMBER"
        try:
            vocab[tok]
        except KeyError:
            u_count += 1.0
            tok = "UNK"
        new_tokens.append(tok)
    return " ".join(new_tokens), t_count, u_count


def write_docs(docdict, wvocab, rvocab, outfname, is_trnfile=False):
    print "Write docs into file: {}".format(outfname)
    if is_trnfile:
        w2vfname = outfname.replace("txt", "w2v")
        print "Write tokens into file: {}".format(w2vfname)
        fw2v = open(w2vfname, 'w')
    total_count, unk_count = 0.0, 0.0
    with open(outfname, 'w') as fout:
        fout.write("EIDX\tPIDX\tRIDX\tEDU\n")
        for (fname, doc) in docdict.iteritems():
            edus = doc.edus
            for eidx in range(len(edus)):
                edu = edus[eidx+1]
                edu, tc, uc = refine_with_vocab(edu, wvocab)
                total_count += tc
                unk_count += uc
                try:
                    ridx = rvocab[doc.relas[eidx+1]]
                except KeyError:
                    ridx = rvocab['elaboration']
                try:
                    pidx = doc.pnodes[eidx+1]-1
                except KeyError:
                    print fname, eidx + 1
                    print doc.pnodes
                line = "{}\t{}\t{}\t{}\n".format(eidx, pidx, ridx, edu)
                fout.write(line)
                # fw2v.write("<s> {} </s>\n".format(edu))
                if is_trnfile:
                    fw2v.write("{}\n".format(edu))
            fout.write("=============\t{}\t{}\n".format(fname, doc.label))
    # counts
    print "Total tokens: {}; UNK counts: {}; Ratio: {}".format(total_count, unk_count, (unk_count/total_count))
    # write vocab
    if is_trnfile:
        fw2v.close()
        vocabfname = outfname.replace("txt", "vocab")
        with open(vocabfname, "w") as fout:
            for (key, val) in wvocab.iteritems():
                fout.write("{}\n".format(key))
        print "Write vocab into file: {}".format(vocabfname)


def main():
    # pls change T and FOLDER at the sametime
    T = 10000
    FOLDER = "fortextclass25-10K"
    SUFFIX = ".brackets25"
    # load labels
    trn_labels = load_labels("../data/yelp/train.labels.gz")
    tst_labels = load_labels("../data/yelp/test.labels.gz")
    # load all files
    rpath = "../data/yelp/parses/"
    flist = get_allbracketsfiles(rpath, SUFFIX)
    trn_docdict, tst_docdict = get_docdict(flist, trn_labels, tst_labels, SUFFIX)
    # get vocabs
    wvocab, rvocab = get_vocab(trn_docdict, thresh=T)
    # write files
    ftrn = "../data/yelp/{}/trn-yelp.txt".format(FOLDER)
    write_docs(trn_docdict, wvocab, rvocab, ftrn, is_trnfile=True)
    ftst = "../data/yelp/{}/tst-yelp.txt".format(FOLDER)
    write_docs(tst_docdict, wvocab, rvocab, ftst)
    infofname = "../data/yelp/{}/info-yelp.txt".format(FOLDER)
    with open(infofname, 'w') as fout:
        fout.write("Size of the training examples: {}\n".format(len(trn_docdict)))
        fout.write("Size of the test examples: {}\n".format(len(tst_docdict)))
        fout.write("Size of the word vocab: {}\n".format(len(wvocab)))
        fout.write("Size of the relation vocab: {}\n".format(len(rvocab)))
    


if __name__ == '__main__':
    main()
