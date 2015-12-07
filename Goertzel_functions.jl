module Goertzel_functions
	export goertzel, online_variance
	function goertzel(data, Nbins, q1o, q2o)
		numSamples = length(data)
		psd = zeros(Nbins+1)
		q1out = copy(q1o) #zeros(Nbins+1)
		q2out = copy(q2o) #zeros(Nbins+1)
		for k=0:Nbins	
			omega = 2.0 * pi * float( float(k)/float(2*(Nbins)) )
			coeff = 2.0 * float(cos(omega))
			q0 = 0.0
			q1 = q1o[k+1]
			q2 = q2o[k+1]
			for i=1:Int64(numSamples)
				q0 = (coeff * q1) - q2 + float(data[i])
				q2 = q1
				q1 = q0
			end
			psd[k+1] = ( (q1^2.0) + (q2^2.0) - (q1 * q2 * coeff) )/numSamples
			q1out[k+1] = q1
			q2out[k+1] = q2
		end
		return psd, q1out, q2out
	end

	function online_variance(data, n0, M1, M2)
		n = n0
		m1 = M1
		m2 = M2
		for x in data
			n += 1
			delta = x - m1
			m1 += delta/n
			m2 += delta*(x - m1)
		end
		if n<2
			return oftype(1.0,NaN), n, m1, m2
		else
			return m2/(n-1), n, m1, m2
		end
	end
end