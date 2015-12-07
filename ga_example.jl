include("reinit_DummyTest.jl")

Npopulations = 1024
Nnumbers = 1024
Nbins = 16

g = GAmodels(128,64,8);
g.par_evaluateAll();
k=1;
while (g.getBest()[2] > 1.0f0 )
	a,b=g.elitistSelection(); 
	g.elitistOrderedCrossOver(a,b);
	g.elitistExchangeMutation(a,b,1.0);
	g.par_evaluateAll();
	println("k = $k  g.getBest() = $(g.getBest())" )
	k+=1
end
