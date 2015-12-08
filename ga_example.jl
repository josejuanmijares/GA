include("reinit_DummyTest.jl")

Npopulations = 1024
Nnumbers = 1024
Nbins = 16

g = GAmodels(128,32,8);
g.par_evaluateAll();
k=1
samevalue_cnt=0
old_best = g.getBest()[2] 
while (g.getBest()[2] > 0.50f0 )
	a,b=g.elitistSelection(); 
	g.elitistOrderedCrossOver(a,b);
	g.elitistExchangeMutation(a,b,1.0);
	g.par_evaluateAll();
	println("k = $k  g.getBest() = $(g.getBest()) old_best = $old_best, cmp = $(g.getBest()[2] == old_best)" )
	
	if (g.getBest()[2] == old_best)
		samevalue_cnt+=1
		
		if samevalue_cnt >10
			#g.elitistOrderedCrossOver(a,b);
			while (g.getBest()[2] == old_best)
				a,b=g.elitistSelection(); 
				g.elitistOrderedCrossOver(a,b);
				g.elitistExchangeMutation(a,b,1.0);
				g.par_evaluateAll();
				println("==== k = $k  g.getBest() = $(g.getBest()) old_best = $old_best, cmp = $(g.getBest()[2] == old_best)" )
			end
			samevalue_cnt=0
			old_best=g.getBest()[2] 
		end
	else
		samevalue_cnt=0
		old_best=g.getBest()[2] 
	end

	
	k+=1
end
