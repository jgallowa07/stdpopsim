"""
A. thaliana msprime example
"""
import msprime

from stdpopsim import arabidopsis_thaliana

chrom = arabidopsis_thaliana.genome.chromosomes["chr5"]
recomb_map = chrom.recombination_map()

model = arabidopsis_thaliana.Durvasula2017MSMC()
model.debug()

samples = [
    msprime.Sample(population=0, time=0), msprime.Sample(population=0, time=0)]

ts = msprime.simulate(
    samples=samples,
    recombination_map=chrom.recombination_map(),
    mutation_rate=chrom.mean_mutation_rate,
    **model.asdict())
print("simulated:", ts.num_trees, ts.num_sites)
