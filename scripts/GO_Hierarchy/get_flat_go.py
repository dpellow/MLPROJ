###
# Download the GO database and gene2go annotations
# Write out file with the GO hierarchy for human:
# <child> \t <parent> \t <child-parent link>
# child to parent link can be: gene, is_a, part_of, [positively_/negatively_]regulates
###

from goatools import obo_parser
from  goatools.associations import read_ncbi_gene2go
from goatools.base import download_ncbi_associations

import wget, os

##
# CONSTANTS: CHANGE HERE IF DIFFERENT FILE PATHS
##
obo_dir = '../../data/'
obo_fname = 'go-basic.obo'
obo_url = 'http://geneontology.org/ontology/go-basic.obo'
out_dir = '../../data/'
out_fname = 'go_flat.txt'
gene2go_dir = '../../data'
gene2go_fname = 'gene2go'

print "Downloading GO obo file"
if not os.path.exists(obo_dir):
        os.makedirs(obo_dir)

if not os.path.exists(os.path.join(obo_dir, obo_fname)):
        wget.download(obo_url, os.path.join(obo_dir,obo_fname))

go_obo = os.path.join(obo_dir, obo_fname)
go = obo_parser.GODag(go_obo,optional_attrs=['relationship']) # also use

print "Downloading gene-GO associations"
if not os.path.exists(gene2go_dir):
        os.makedirs(gene2go_dir)
        
gene2go = download_ncbi_associations(os.path.join(gene2go_dir,gene2go_fname))
go2geneids_human = read_ncbi_gene2go(os.path.join(gene2go_dir,gene2go_fname), taxids=[9606], go2geneids=True)


print "Writing out GO child-parent links"
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

with open(os.path.join(out_dir,out_fname),'w') as o:
    for goid in go2geneids_human.keys():
        entry = go[goid]
        for gene in go2geneids_human[entry.id]:
            o.write(str(gene) + '\t' + entry.id + '\t' + 'gene' + '\n')
        children = entry.children
        for c in children:
            o.write(c.id + '\t' + entry.id + '\t' + 'is_a' + '\n')
        rels = entry.relationship_rev
        for rtype in rels.keys():
            rs = rels[rtype]
            for r in rs:
                o.write(r.id + '\t' + entry.id + '\t' + rtype + '\n')
