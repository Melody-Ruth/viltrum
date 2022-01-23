#pragma once
#include <iostream>
#include <unordered_map>
using namespace std;
#include <vector>
#include "integrate.h"
#include "multidimensional-range.h"
#include <random>
#include <vector>
#include <array>
#include <type_traits>

namespace viltrum {

template<typename Nested, typename Error>
class RegionGenerator {
	StepperAdaptive<Nested,Error> stepper;
	unsigned long adaptive_iterations;
public:
    RegionGenerator(Nested&& nested, Error&& error, unsigned long ai) :
        stepper(std::forward<Nested>(nested),std::forward<Error>(error)),
        adaptive_iterations(ai) { }
		
	template<typename F, typename Float, std::size_t DIM>
    auto compute_regions(const F& f, const Range<Float,DIM>& range) const {
		//cout << "Computing regions!" << endl;
        auto regions = stepper.init(f,range);
        for (unsigned long i = 0; i<adaptive_iterations; ++i) {
            stepper.step(f,range,regions);
		}

		cout << "Region results: " << endl;
		//cout << regions.size() << endl;
		//int numRegions = 0;
		for (auto region : regions) {
			//numRegions++;
			//cout << region.range().min()[0] << ", " << region.range().max()[0] << " and ";
			//cout << region.range().min()[1] << ", " << region.range().max()[1] << endl;
		}
        return regions;
    }
};

template<typename Nested, typename Error>
auto region_generator(Nested&& nested, Error&& error, unsigned long adaptive_iterations) {
	return RegionGenerator<std::decay_t<Nested>, std::decay_t<Error>>(
		std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations);
}

template<typename Float, std::size_t DIM, std::size_t DIMBINS, typename R>
auto pixels_in_region(const R& r, const std::array<std::size_t,DIMBINS>& bin_resolution, const Range<Float,DIM>& range) {
	//range is the entire integration domain, r is the particular bin we're looking at
	std::array<std::size_t,DIMBINS> start_bin, end_bin;
    for (std::size_t i = 0; i<DIMBINS;++i) {
        start_bin[i] = std::max(std::size_t(0),std::size_t(Float(bin_resolution[i])*(r.range().min(i) - range.min(i))/(range.max(i) - range.min(i))));
        end_bin[i]   = std::max(start_bin[i]+1,std::min(bin_resolution[i],std::size_t(0.99f + 
                    (Float(bin_resolution[i])*(r.range().max(i) - range.min(i))/(range.max(i) - range.min(i))))));
    }
	return multidimensional_range(start_bin,end_bin);
}

class AlphaOptimized {
	double alpha_min, alpha_max;
	
    template<typename T>
    T alpha(const T& cov, const T& var, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0) const {
		return (var==T(0))?((cov>T(0))?T(0):T(alpha_max)):std::max(T(alpha_min),std::min(T(alpha_max),(cov/var)));
	}

	template<typename T>
	T alpha(const T& cov, const T& var, typename std::enable_if<!std::is_floating_point<T>::value>::type* = 0) const {
		T res;
		for (decltype(res.size()) i = 0; i<res.size(); i++) {
			res[i] = alpha(cov[i], var[i]);
		}
		return res;
	}
public:
	AlphaOptimized(double amin = -1.0, double amax = 1.0) : alpha_min(amin), alpha_max(amax) { }
	
	template<typename T>
	T alpha(const std::vector<std::tuple<T,T>>& samples) const {
		if (samples.size()<=1) return T(1);
		T mean_f, mean_app;
		std::tie(mean_f,mean_app) = samples[0];
		for (std::size_t s=1; s<samples.size();++s) {
			mean_f += std::get<0>(samples[s]); mean_app += std::get<1>(samples[s]);		
		}
		mean_f/=double(samples.size());  	mean_app/=double(samples.size());
				
		T cov = (std::get<0>(samples[0]) - mean_f)*(std::get<1>(samples[0]) - mean_app);
		T var_app = (std::get<1>(samples[0]) - mean_app)*(std::get<1>(samples[0]) - mean_app);
		for (std::size_t s=1; s<samples.size();++s) {
			cov += (std::get<0>(samples[s]) - mean_f)*(std::get<1>(samples[s]) - mean_app);
			var_app += (std::get<1>(samples[s]) - mean_app)*(std::get<1>(samples[s]) - mean_app);		
		}
//		cov/=double(samples.size()); var_app/=double(samples.size());
//		^^ unnecesary division as it gets simplified afterwards 
		return alpha(cov,var_app);
	}
};

class AlphaConstant {
	double value;
public:
	AlphaConstant(double a=1.0) : value(a) { }
	
	template<typename T>
	T alpha(const std::vector<std::tuple<T,T>>& samples) const {
		return T(value);
	}
};

class FunctionSampler {
public:
	template<typename F, typename Float, std::size_t DIM, typename RNG>
	auto sample(const F& f, const Range<Float,DIM>& range, RNG& rng) const {
		std::array<Float,DIM> sample;
	    for (std::size_t i=0;i<DIM;++i) {
		    std::uniform_real_distribution<Float> dis(range.min(i),range.max(i));
		    sample[i] = dis(rng);
		}
		return std::make_tuple(f(sample),sample);
	}
};

template<typename Float, std::size_t DIM, std::size_t DIMBINS>
Range<Float,DIMBINS> range_of_pixel(const std::array<std::size_t,DIMBINS>& pixel, const std::array<std::size_t,DIMBINS>& bin_resolution, const Range<Float,DIM>& range) {
	std::array<Float, DIMBINS> submin, submax;
	for (std::size_t i=0;i<DIMBINS;++i) {
		submin[i] = range.min(i)+pixel[i]*(range.max(i)-range.min(i))/Float(bin_resolution[i]);
		submax[i] = range.min(i)+(pixel[i]+1)*(range.max(i)-range.min(i))/Float(bin_resolution[i]);
	}
	return Range<Float, DIMBINS>(submin,submax); 
}


template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
class IntegratorStratifiedAllControlVariates {
	RegionGenerator region_generator;
	AlphaCalculator alpha_calculator;
	Sampler sampler;
	mutable RNG rng;
	unsigned long spp;
public:

	typedef void is_integrator_tag;

	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins& bins, const std::array<std::size_t,DIMBINS>& bin_resolution, const F& f, const Range<Float,DIM>& range) const {
        auto regions = region_generator.compute_regions(f,range);
		using value_type = decltype(f(range.min()));
		using R = typename decltype(regions)::value_type;
		vector_dimensions<std::vector<const R*>,DIMBINS> regions_per_pixel(bin_resolution);
		for (const auto& r : regions) for (auto pixel : pixels_in_region(r,bin_resolution,range))
			regions_per_pixel[pixel].push_back(&r);
		for (auto pixel : multidimensional_range(bin_resolution)) { // Per pixel
			bins(pixel) = value_type(0);
			auto pixel_range = range_of_pixel(pixel,bin_resolution,range);
			const auto& regions_here = regions_per_pixel[pixel];
			std::size_t samples_per_region = spp / regions_here.size();
            std::size_t samples_per_region_rest = spp % regions_here.size();
			std::uniform_int_distribution<std::size_t> sr(std::size_t(0),regions_here.size()-1);
			std::size_t sampled_region = sr(rng);
			for (std::size_t r = 0; r<regions_here.size(); ++r) { // Per region inside the pixel
                std::size_t nsamples = samples_per_region;
				if (((r - sampled_region)%(regions_here.size()))<samples_per_region_rest) nsamples+=1;
                auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				if (nsamples == 0) 
					bins(pixel) += double(regions_per_pixel.size())*regions_here[r]->integral_subrange(local_range);
				else {
					std::vector<std::tuple<value_type,value_type>> samples(nsamples);
				    double factor = local_range.volume()*double(regions_per_pixel.size())*double(regions_here.size());
				//  ^^ MC probability      ^^ Global size (res-constant)     ^^ Region probability
		
					for (auto& s : samples) {
						auto [value,sample] = sampler.sample(f,local_range,rng);
						s = std::make_tuple(factor*value, factor*regions_here[r]->approximation_at(sample));
					}
					auto a = alpha_calculator.alpha(samples);
					value_type residual = (std::get<0>(samples[0]) - a*std::get<1>(samples[0]));
					for (std::size_t s=1; s<nsamples;++s)
						residual += (std::get<0>(samples[s]) - a*std::get<1>(samples[s]));
				
					//We are multiplying all the samples by the number of regions so we do this
					bins(pixel) += (residual/double(spp)); //... instead of this -> (residual/double(nsamples))
					bins(pixel) += double(regions_per_pixel.size())*a*regions_here[r]->integral_subrange(local_range);
					//If we covered each region independently we would not multiply by the number of regions
				}
			}
		}
	}
	
	IntegratorStratifiedAllControlVariates(RegionGenerator&& region_generator,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) :
			region_generator(std::forward<RegionGenerator>(region_generator)),
			alpha_calculator(std::forward<AlphaCalculator>(alpha_calculator)),
			sampler(std::forward<Sampler>(sampler)),
			rng(std::forward<RNG>(rng)), spp(spp) {}
};

template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
class IntegratorStratifiedPixelControlVariates {
	RegionGenerator region_generator;
	AlphaCalculator alpha_calculator;
	Sampler sampler;
	mutable RNG rng;
	unsigned long spp;
public:

	typedef void is_integrator_tag;

	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins& bins, const std::array<std::size_t,DIMBINS>& bin_resolution, const F& f, const Range<Float,DIM>& range) const {
		std::array<Float,DIM> graphingSample;
		int numZero = 0;
		int numTotal = 0;
		int totalOther = 0;
		int totalFive = 0;
		int completeTotal = 0;
		/*for (double i = 0; i < 1; i+=0.1) {
			for (double j = 0; j < 1; j+= 0.1) {
				for (double k = 0; k < 1; k+=0.1) {
					for (double l = 0; l < 1; l+= 0.1) {
						graphingSample[0] = 0.833705;
						graphingSample[1] = 0.347211;
						graphingSample[2] = j;
						graphingSample[4] = l;
						graphingSample[2] = i;
						graphingSample[3] = k;
						//0.833705,0.347211
						numTotal++;
						auto [tempFValue] = f(graphingSample, rng, true);
						//cout << "Result of graphing that: " << endl;
						//cout << tempFValue[0] << ", " << tempFValue[1] << ", " << tempFValue[2] << endl;
						//numZero++;
						if (tempFValue[0] != 0 && (tempFValue[0] < 4.9 || tempFValue[0] > 5.1)) {
							cout << i << "," << k << "," << tempFValue[0] << endl;
							cout << "Hello\n";
						}
					}
				}
			}
		}*/
		//cout << "Tested " << numTotal << "points\n";
		//cout << "Of those, " << (numTotal-numZero) << " had non-zero values\n";

		cout << "Bin dimensions: " << DIMBINS << endl;
		for (int i = 0; i < DIM; i++) {
			cout << range.min(i) << ", " << range.max(i) << endl;
		}
        auto regions = region_generator.compute_regions(f,range);
		using value_type = decltype(f(range.min()));
		using R = typename decltype(regions)::value_type;
		vector_dimensions<std::vector<const R*>,DIMBINS> regions_per_pixel(bin_resolution);
		int tempCount = 0;
		for (const auto& r : regions) {
			tempCount++;
		}
		cout << "Number of regions total: " << tempCount << endl;
		tempCount = 0;
		for (auto pixel : multidimensional_range(bin_resolution)) {
			tempCount++;
		}
		cout << f(range.min())[0] << ", " << f(range.min())[1] << ", " << f(range.min())[2] << endl;
		cout << "Number of pixels total: " << tempCount << endl;
		for (const auto& r : regions) for (auto pixel : pixels_in_region(r,bin_resolution,range))
			regions_per_pixel[pixel].push_back(&r);
		int totalNumZero = 0;
		int totalNumTotal = 0;
		//cout << "Hi!!!!!\n";
		for (auto pixel : multidimensional_range(bin_resolution)) { // Per pixel
			//cout << "Hello1\n";
			totalNumTotal++;
			bins(pixel) = value_type(0);
			auto pixel_range = range_of_pixel(pixel,bin_resolution,range);
			int numSamples = 0;
			double estimate = 0;
			//if (pixel[0] == 283 && pixel[1] == 176) {
			/*if (pixel[0] == 203 && pixel[1] == 106) {
				cout << "Hello??? It's me" << endl;
				cout << "i range: " << pixel_range.min(0) << " to " << pixel_range.max(0) << endl;
				cout << "j range: " << pixel_range.min(1) << " to " << pixel_range.max(1) << endl;
				cout << "k range: " << pixel_range.min(2) << " to " << pixel_range.max(2) << endl;
				cout << "l range: " << pixel_range.min(3) << " to " << pixel_range.max(3) << endl;
				for (double i = pixel_range.min(0); i < pixel_range.max(0); i+=0.00005) {
					//cout << "Hello, i = " << i << endl;
					for (double j = pixel_range.min(1); j < pixel_range.max(1); j+=0.00005) {
						//cout << "Hello, j = " << j << endl;
						for (double k = 0; k < 1; k+=0.01) {
							for (double l = 0; l < 1; l+=0.01) {
								numSamples++;
								graphingSample[0] = i;
								graphingSample[1] = j;
								graphingSample[2] = k;
								graphingSample[3] = l;
								auto [tempFValue] = f(graphingSample, rng, true);
								if (tempFValue[0] != 0) {
									cout << "Graphing sample: " << endl;
									cout << graphingSample[0] << endl;
									cout << graphingSample[1] << endl;
									cout << graphingSample[2] << endl;
									cout << graphingSample[3] << endl;
									
									cout << "Result of graphing that: " << endl;
									for (auto tempFPart : tempFValue) {
										cout << tempFPart << endl;
									}
								}
								estimate += tempFValue[0];
								
								//cout << i << ", " << tempFValue[0] << ", " << tempFValue[1] << ", " << tempFValue[2] << endl;
							}
						}
					}
				}
				estimate = estimate/numSamples;
				cout << "Estimate: " << estimate << " (no adjustment for size) " << endl;
				//estimate *= local_range.volume()*double(regions_per_pixel.size())*double(regions_here.size())
				estimate *= pixel_range.max(0)-pixel_range.min(0);
				estimate *= pixel_range.max(1)-pixel_range.min(1);
				cout << "Actually, probably " << estimate << endl;
			}*/
			const auto& regions_here = regions_per_pixel[pixel];
			std::size_t samples_per_region = spp / regions_here.size();
			//cout << "Hello???" << endl;
			//cout << "Number of regions associated with this pixel: " << regions_here.size() << ", spp: " << spp << ", samples per region: " << samples_per_region << endl;
			//cout << samples_per_region << endl;
            std::size_t samples_per_region_rest = spp % regions_here.size();
			std::vector<std::tuple<value_type,value_type>> samples; samples.reserve(spp);
			
            //First: stratified distribution of samples (uniformly)
			for (std::size_t r = 0; r<regions_here.size(); ++r) { 
                auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				double factor = local_range.volume()*double(regions_per_pixel.size())*double(regions_here.size());
				for (std::size_t s = 0; s<samples_per_region; ++s) {
					auto [value,sample] = sampler.sample(f,local_range,rng);
					samples.push_back(std::make_tuple(factor*value, factor*regions_here[r]->approximation_at(sample)));
					//if (pixel[0] == 283 && pixel[1] == 176) {
					//if (regions_here[r]->approximation_at(sample)[1] > 0) {
						//cout << "Approximation: " << regions_here[r]->approximation_at(sample)[1] << endl;
					//}
					completeTotal++;
					if (value[0] == 0) {
						numZero++;
					} else if (value[0] > 4.99 && value[0] < 5.01) {
						totalFive++;
					} else {
						totalOther++;
					}
					//if (value[0] > 0) {
						//cout << "Found: \n";
						//cout << sample[0] << "," << sample[1] << "," << sample[2] << "," << sample[3] << endl;
					//}
					//if (pixel[0] > 282 && pixel[0] < 284 && pixel[1] > 175 && pixel[1] < 177 && value[0] != 0) {
					//if (pixel[0] > 282 && pixel[1] < 177 && value[0] != 0) {
					//if (pixel[0] > 250 && pixel[0] < 325 && pixel[1] > 125 && pixel[1] < 225 && value[0] != 0) {
						//cout << pixel[0] << ", " << pixel[1] << endl;
						//cout << "Found it!!!\n";
						//cout << "Not 0! " << value[0] << endl;
					//}
					/*if (pixel[0] > 280 && pixel[0] < 285 && pixel[1] > 170 && pixel[1] < 180 && value[0] != 0) {
						cout << pixel[0] << ", " << pixel[1] << endl;
						cout << "Not 0 again! " << value[0] << endl;
					}*/
				}
			} 
			//cout << "Hello2\n";
            std::uniform_int_distribution<std::size_t> sample_region(std::size_t(0),regions_here.size()-1);
            //We randomly distribute the rest of samples among all regions 
			int numZero = 0;
			int numFive = 0;
			int numOther = 0;
            for (std::size_t i = 0; i<samples_per_region_rest; ++i) {
			    std::size_t r = sample_region(rng);
                auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				double factor = local_range.volume()*double(regions_per_pixel.size())*double(regions_here.size());
				auto [value,sample] = sampler.sample(f,local_range,rng);
				samples.push_back(std::make_tuple(factor*value, factor*regions_here[r]->approximation_at(sample)));
				//if (pixel[0] == 283 && pixel[1] == 176) {
					//cout << "Approximation: " << regions_here[r]->approximation_at(sample)[0] << endl;
				//}
				/*if (value[0] == 0) {
					numZero++;
				} else if (value[0] > 4.99 && value[0] < 5.01) {
					numFive++;
					totalFive++;
				} else {
					numOther++;
					totalOther++;
				}
				completeTotal++;*/
				/*if (value[0] != 0 && (value[0] < 4.99 || value[0] > 5.01)) {
					cout << value[0] << " versus " << (factor*value)[0] << endl;
				}*/
				
            }
			//if (numOther > 0) {
				//cout << numZero << " zeros, " << numFive << " fives, and " << numOther << " that are neither 0 nor 5\n";
			//}
			

			double actualEstimate = 0;
			double actualEstimate2 = 0;
			auto a = alpha_calculator.alpha(samples);
			//cout << "Hello???" << endl;
			//cout << sampler << endl;
			value_type residual = (std::get<0>(samples[0]) - a*std::get<1>(samples[0]));
			numTotal = 0;
			numZero = 0;
			int numDiff = 0;
			for (std::size_t s=1; s<spp;++s) {
				residual += (std::get<0>(samples[s]) - a*std::get<1>(samples[s]));
				if (pixel[0] == 283 && pixel[1] == 176) {
					//cout << std::get<1>(samples[s])[0] << endl;
				}
				if (pixel[0] == 203 && pixel[1] == 106) {
					cout << std::get<1>(samples[s])[0] << endl;
				}
				actualEstimate += std::get<1>(samples[s])[0];
				actualEstimate2 += std::get<1>(samples[s])[2];
				if (std::get<0>(samples[s])[0] != 0 && std::get<0>(samples[s])[0] != 5) {
					numDiff++;
				}
				numTotal++;
				if (std::get<0>(samples[s])[0] == 0) {
					numZero++;
				} else {
					//cout << std::get<0>(samples[s])[0] << endl;
				}
			}

			if (pixel[0] == 283 && pixel[1] == 176) {
				//cout << "283 one\n";
				//cout << "Their estimate(?): " << actualEstimate << endl;
				//cout << "residual: " << residual[0]/double(spp) << endl;
			}
			if (pixel[0] == 203 && pixel[1] == 106) {
				cout << "Red???" << endl;
				cout << "Their estimate(?): " << actualEstimate << endl;
				cout << "Another possibility: " << actualEstimate/double(spp) << endl;
				cout << "residual: " << residual[0]/double(spp) << endl;
				cout << "Blue??" << endl;
				cout << "Their estimate(?): " << actualEstimate2 << endl;
				cout << "Another possibility: " << actualEstimate2/double(spp) << endl;
				cout << "residual: " << residual[2]/double(spp) << endl;
			}
			if (numZero == numTotal) {
				totalNumZero++;
				//cout << pixel[0] << ", " << pixel[1] << endl;
			}
			//cout << "Tested " << numTotal << "points\n";
			//cout << "Of those, " << (numTotal-numZero) << " had non-zero values\n";
			//cout << "Of those, " << numDiff << " had a value besides 0 or 5\n";
			bins(pixel) += (residual/double(spp));
			for (auto r : regions_here) bins(pixel) += double(regions_per_pixel.size())*a*r->integral_subrange(pixel_range.intersection_large(r->range()));
		}
		cout << "Tested " << totalNumTotal << "pixels\n";
		cout << "Of those, " << (totalNumTotal - totalNumZero) << " had non-zero values\n";
		cout << (completeTotal-totalOther-totalFive) << " zeros, " << totalFive << " fives, and " << totalOther << " that are neither 0 nor 5\n";
		//cout << totalOther << " are not 5 or 0 of " << completeTotal << endl;
	}
	
	IntegratorStratifiedPixelControlVariates(RegionGenerator&& region_generator,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) :
			region_generator(std::forward<RegionGenerator>(region_generator)),
			alpha_calculator(std::forward<AlphaCalculator>(alpha_calculator)),
			sampler(std::forward<Sampler>(sampler)),
			rng(std::forward<RNG>(rng)), spp(spp) {}
};

template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
class IntegratorStratifiedPixelControlVariatesAntithetic {
	RegionGenerator region_generator;
	AlphaCalculator alpha_calculator;
	Sampler sampler;
	mutable RNG rng;
	unsigned long spp;
public:

	typedef void is_integrator_tag;

	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins& bins, const std::array<std::size_t,DIMBINS>& bin_resolution, const F& f, const Range<Float,DIM>& range) const {
        auto regions = region_generator.compute_regions(f,range);
		using value_type = decltype(f(range.min()));
		using R = typename decltype(regions)::value_type;
		vector_dimensions<std::vector<const R*>,DIMBINS> regions_per_pixel(bin_resolution);
		unordered_map<int, int> samplesPerRegionCount;
		for (const auto& r : regions) for (auto pixel : pixels_in_region(r,bin_resolution,range))
			regions_per_pixel[pixel].push_back(&r);
		for (auto pixel : multidimensional_range(bin_resolution)) { // Per pixel
			bins(pixel) = value_type(0);
			auto pixel_range = range_of_pixel(pixel,bin_resolution,range);
			const auto& regions_here = regions_per_pixel[pixel];
			std::size_t samples_per_region = spp / regions_here.size();
			//cout << samples_per_region << endl;
			/*if (samplesPerRegionCount.find(samples_per_region) != samplesPerRegionCount.end()) {
				samplesPerRegionCount[samples_per_region] = samplesPerRegionCount[samples_per_region]+1;
			} else {
				samplesPerRegionCount[samples_per_region] = 1;
			}*/
            std::size_t samples_per_region_rest = spp % regions_here.size();
			std::vector<std::tuple<value_type,value_type>> samples; samples.reserve(spp);
			
            //First: stratified distribution of samples (uniformly)
			for (std::size_t r = 0; r<regions_here.size(); ++r) { 
                auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				double factor = local_range.volume()*double(regions_per_pixel.size())*double(regions_here.size());
				for (std::size_t s = 0; s<samples_per_region/2; ++s) {
					auto [value,sample] = sampler.sample(f,local_range,rng);
					samples.push_back(std::make_tuple(factor*value, factor*regions_here[r]->approximation_at(sample)));
					auto [value2,sample2] = sampler.sampleOpposite(f,local_range,rng,sample);
					samples.push_back(std::make_tuple(factor*value2, factor*regions_here[r]->approximation_at(sample2)));

				}
				if (samples_per_region % 2 != 0) {
					//non-antithetic extra
					auto [value,sample] = sampler.sample(f,local_range,rng);
					samples.push_back(std::make_tuple(factor*value, factor*regions_here[r]->approximation_at(sample)));
				}
			} 
            std::uniform_int_distribution<std::size_t> sample_region(std::size_t(0),regions_here.size()-1);
            //We randomly distribute the rest of samples among all regions 
            for (std::size_t i = 0; i<samples_per_region_rest; ++i) {
			    std::size_t r = sample_region(rng);
                auto local_range = pixel_range.intersection_large(regions_here[r]->range());
				double factor = local_range.volume()*double(regions_per_pixel.size())*double(regions_here.size());
				auto [value,sample] = sampler.sample(f,local_range,rng);
				samples.push_back(std::make_tuple(factor*value, factor*regions_here[r]->approximation_at(sample)));
            }


			auto a = alpha_calculator.alpha(samples);
			//cout << "Hello???" << endl;
			//cout << sampler << endl;
			value_type residual = (std::get<0>(samples[0]) - a*std::get<1>(samples[0]));
			for (std::size_t s=1; s<spp;++s)
				residual += (std::get<0>(samples[s]) - a*std::get<1>(samples[s]));
			bins(pixel) += (residual/double(spp));
			for (auto r : regions_here) bins(pixel) += double(regions_per_pixel.size())*a*r->integral_subrange(pixel_range.intersection_large(r->range()));
		}
		//cout << "Samples per regions, count\n";
		/*for (auto myPair : samplesPerRegionCount) {
			cout << x.first << ", " << x.second << endl;
		}*/
	}
	
	IntegratorStratifiedPixelControlVariatesAntithetic(RegionGenerator&& region_generator,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) :
			region_generator(std::forward<RegionGenerator>(region_generator)),
			alpha_calculator(std::forward<AlphaCalculator>(alpha_calculator)),
			sampler(std::forward<Sampler>(sampler)),
			rng(std::forward<RNG>(rng)), spp(spp) {}
};

template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
class IntegratorStratifiedRegionControlVariates {
	RegionGenerator region_generator;
	AlphaCalculator alpha_calculator;
	Sampler sampler;
	mutable RNG rng;
	unsigned long spp;
public:

	typedef void is_integrator_tag;

	template<typename Bins, std::size_t DIMBINS, typename F, typename Float, std::size_t DIM>
	void integrate(Bins& bins, const std::array<std::size_t,DIMBINS>& bin_resolution, const F& f, const Range<Float,DIM>& range) const {
        auto regions = region_generator.compute_regions(f,range);
		using value_type = decltype(f(range.min()));
		
		auto all_pixels = multidimensional_range(bin_resolution);
		
		for (auto pixel : all_pixels) bins(pixel) = value_type(0);
		
		std::size_t samples_per_region = (spp*all_pixels) / regions.size();
        std::size_t samples_per_region_rest = (spp*all_pixels) % regions.size();
		std::uniform_int_distribution<std::size_t> sr(std::size_t(0),regions.size()-1);
		std::size_t sampled_region = sr(rng);
		
//		std::cerr<<"Regions = "<<regions.size()<<" - Samples per region "<<samples_per_region<<" - Rest = "<<samples_per_region_rest<<std::endl;
		
		std::size_t ri=0;
		for (const auto& r : regions) {
			std::size_t nsamples = samples_per_region + 
				(( (((ri++) + sampled_region) % regions.size()) < samples_per_region_rest )?1:0);
			auto pixels = pixels_in_region(r,bin_resolution,range);
			
			std::size_t samples_per_pixel = nsamples / pixels;
			std::size_t samples_per_pixel_rest = nsamples % pixels;
			
//			std::cerr<<"    Region "<<ri<<" -> "<<nsamples<<" samples, "<<int(pixels)<<" pixels, "<<samples_per_pixel<<" samples per pixel, "<<samples_per_pixel_rest<<" random samples"<<std::endl;
			
			std::vector<std::tuple<value_type,value_type>> samples; samples.reserve(nsamples);
			std::vector<std::array<std::size_t,DIMBINS>> positions; positions.reserve(nsamples);
			
			for (auto pixel : pixels) {
				auto pixel_range = range_of_pixel(pixel,bin_resolution,range).intersection_large(r.range());
			    double factor = pixel_range.volume()*double(all_pixels)*double(pixels);
				for (std::size_t pass = 0; pass < samples_per_pixel; ++pass) {
					auto [value,sample] = sampler.sample(f,pixel_range,rng);
					samples.push_back(std::make_tuple(factor*value, factor*r.approximation_at(sample)));
					positions.push_back(pixel);
				}
			}
			
			double factor = r.range().volume()*double(all_pixels);
			for (std::size_t i = 0; i < samples_per_pixel_rest; ++i) {
				auto [value,sample] = sampler.sample(f,r.range(),rng);
				samples.push_back(std::make_tuple(factor*value, factor*r.approximation_at(sample)));
				std::array<std::size_t,DIMBINS> pixel;
				for (std::size_t i=0;i<DIMBINS;++i) 
					pixel[i] = std::size_t(bin_resolution[i]*(sample[i] - range.min(i))/(range.max(i) - range.min(i)));
				positions.push_back(pixel);
            }
			
			auto a = alpha_calculator.alpha(samples);
			double residual_factor = 1.0/double(nsamples);
			
			for (std::size_t s=0; s<nsamples;++s) {
				bins(positions[s]) += residual_factor*(std::get<0>(samples[s]) - a*std::get<1>(samples[s]));
			}
			for (auto pixel : pixels) {
				auto pixel_range = range_of_pixel(pixel,bin_resolution,range).intersection_large(r.range());
				bins(pixel) += (a*double(all_pixels))*r.integral_subrange(pixel_range);
			}
		}
	}
	
	IntegratorStratifiedRegionControlVariates(RegionGenerator&& region_generator,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) :
			region_generator(std::forward<RegionGenerator>(region_generator)),
			alpha_calculator(std::forward<AlphaCalculator>(alpha_calculator)),
			sampler(std::forward<Sampler>(sampler)),
			rng(std::forward<RNG>(rng)), spp(spp) {}
};


template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
auto integrator_stratified_all_control_variates(RegionGenerator&& rg,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) {
	return IntegratorStratifiedAllControlVariates<
		std::decay_t<RegionGenerator>,std::decay_t<AlphaCalculator>,std::decay_t<Sampler>,std::decay_t<RNG>>(
			std::forward<RegionGenerator>(rg),
			std::forward<AlphaCalculator>(alpha_calculator),
			std::forward<Sampler>(sampler),
			std::forward<RNG>(rng),
			spp);
}

template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
auto integrator_stratified_pixel_control_variates(RegionGenerator&& rg,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) {
	return IntegratorStratifiedPixelControlVariates<
		std::decay_t<RegionGenerator>,std::decay_t<AlphaCalculator>,std::decay_t<Sampler>,std::decay_t<RNG>>(
			std::forward<RegionGenerator>(rg),
			std::forward<AlphaCalculator>(alpha_calculator),
			std::forward<Sampler>(sampler),
			std::forward<RNG>(rng),
			spp);
}

template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
auto integrator_stratified_pixel_control_variates_antithetic(RegionGenerator&& rg,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) {
	return IntegratorStratifiedPixelControlVariatesAntithetic<
		std::decay_t<RegionGenerator>,std::decay_t<AlphaCalculator>,std::decay_t<Sampler>,std::decay_t<RNG>>(
			std::forward<RegionGenerator>(rg),
			std::forward<AlphaCalculator>(alpha_calculator),
			std::forward<Sampler>(sampler),
			std::forward<RNG>(rng),
			spp);
}

template<typename RegionGenerator, typename AlphaCalculator, typename Sampler, typename RNG>
auto integrator_stratified_region_control_variates(RegionGenerator&& rg,
		AlphaCalculator&& alpha_calculator, Sampler&& sampler, RNG&& rng, unsigned long spp) {
	return IntegratorStratifiedRegionControlVariates<
		std::decay_t<RegionGenerator>,std::decay_t<AlphaCalculator>,std::decay_t<Sampler>,std::decay_t<RNG>>(
			std::forward<RegionGenerator>(rg),
			std::forward<AlphaCalculator>(alpha_calculator),
			std::forward<Sampler>(sampler),
			std::forward<RNG>(rng),
			spp);
}


template<typename Nested, typename Error, typename RNG>
auto integrator_optimized_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, RNG&& rng) {
			
	return integrator_stratified_all_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaOptimized(), FunctionSampler(), std::forward<RNG>(rng), spp);
}

template<typename Nested, typename Error, typename RNG>
auto integrator_optimized_perpixel_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, RNG&& rng) {
			
	return integrator_stratified_pixel_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaOptimized(), FunctionSampler(), std::forward<RNG>(rng), spp);
}

template<typename Nested, typename Error, typename RNG>
auto integrator_optimized_perregion_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, RNG&& rng) {
			
	return integrator_stratified_region_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaOptimized(), FunctionSampler(), std::forward<RNG>(rng), spp);
}

template<typename Nested, typename Error, typename RNG>
auto integrator_alpha1_perregion_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, RNG&& rng) {
			
	return integrator_stratified_region_control_variates(region_generator(std::forward<Nested>(nested), std::forward<Error>(error), adaptive_iterations), AlphaConstant(), FunctionSampler(), std::forward<RNG>(rng), spp);
}



template<typename Nested, typename Error>
auto integrator_optimized_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()()) {
		
		return integrator_optimized_adaptive_stratified_control_variates(
		std::forward<Nested>(nested),std::forward<Error>(error),adaptive_iterations, spp, std::mt19937_64(seed));
}

template<typename Nested, typename Error>
auto integrator_optimized_perpixel_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()()) {
			
	return integrator_optimized_perpixel_adaptive_stratified_control_variates(
		std::forward<Nested>(nested),std::forward<Error>(error),adaptive_iterations, spp, std::mt19937_64(seed));
}

template<typename Nested, typename Error>
auto integrator_optimized_perregion_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()()) {
			
	return integrator_optimized_perregion_adaptive_stratified_control_variates(
		std::forward<Nested>(nested),std::forward<Error>(error),adaptive_iterations, spp, std::mt19937_64(seed));
}

template<typename Nested, typename Error>
auto integrator_alpha1_perregion_adaptive_stratified_control_variates(Nested&& nested, Error&& error, 
		unsigned long adaptive_iterations, unsigned long spp, std::size_t seed = std::random_device()()) {
			
	return integrator_alpha1_perregion_adaptive_stratified_control_variates(
		std::forward<Nested>(nested),std::forward<Error>(error),adaptive_iterations, spp, std::mt19937_64(seed));
}

}





