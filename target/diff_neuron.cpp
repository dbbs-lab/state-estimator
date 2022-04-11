/*
*  diff_neuron.cpp
*
*  This file is part of NEST.
*
*  Copyright (C) 2004 The NEST Initiative
*
*  NEST is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 2 of the License, or
*  (at your option) any later version.
*
*  NEST is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
*
*  2021-03-01 11:52:50.594481
*/

// C++ includes:
#include <limits>
#include <cmath>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

#include "diff_neuron.h"


/* ----------------------------------------------------------------
* Recordables map
* ---------------------------------------------------------------- */
nest::RecordablesMap<diff_neuron> diff_neuron::recordablesMap_;

namespace nest
{
  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
  template <> void RecordablesMap<diff_neuron>::create(){
  // use standard names whereever you can for consistency!

  insert_("in_rate", &diff_neuron::get_in_rate);

  insert_("out_rate", &diff_neuron::get_out_rate);
  }
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * Note: the implementation is empty. The initialization is of variables
 * is a part of the diff_neuron's constructor.
 * ---------------------------------------------------------------- */
diff_neuron::Parameters_::Parameters_(){}

diff_neuron::State_::State_(){}

/* ----------------------------------------------------------------
* Parameter and state extractions and manipulation functions
* ---------------------------------------------------------------- */

diff_neuron::Buffers_::Buffers_(diff_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

diff_neuron::Buffers_::Buffers_(const Buffers_ &, diff_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */
diff_neuron::diff_neuron():Archiving_Node(), P_(), S_(), B_(*this)
{
  recordablesMap_.create();

  P_.kp = 1; // as real
  P_.pos = true; // as boolean
  P_.buffer_size = 100*1.0; // as real
  P_.base_rate = 0; // as real
  S_.in_rate = 0; // as real
  S_.out_rate = 0; // as real
}

diff_neuron::diff_neuron(const diff_neuron& __n):
  Archiving_Node(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this){
  P_.kp = __n.P_.kp;
  P_.pos = __n.P_.pos;
  P_.buffer_size = __n.P_.buffer_size;
  P_.base_rate = __n.P_.base_rate;

  S_.in_rate = __n.S_.in_rate;
  S_.out_rate = __n.S_.out_rate;

}

diff_neuron::~diff_neuron(){
}

/* ----------------------------------------------------------------
* Node initialization functions
* ---------------------------------------------------------------- */

void diff_neuron::init_state_(const Node& proto){
  const diff_neuron& pr = downcast<diff_neuron>(proto);
  S_ = pr.S_;
}



void diff_neuron::init_buffers_(){

  get_inh_spikes().clear(); //includes resize
  get_exc_spikes().clear(); //includes resize

  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();

}

void diff_neuron::calibrate(){
  B_.logger_.init();
}

/* ----------------------------------------------------------------
* Update and spike handling functions
* ---------------------------------------------------------------- */

/*
 *
 */
void diff_neuron::update(nest::Time const & origin,const long from, const long to){

  librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );

  long tick = origin.get_steps();
  double time_res = nest::Time::get_resolution().get_ms();

  long buf_sz = std::lrint(P_.buffer_size / time_res);
  double spike_count_in_pos = 0;
  double spike_count_in_neg = 0;
  double spike_count_in = 0;
  long spike_count_out = 0;

  double lamda_est = 0;   // Expected number of excitatory spikes into the buffer
  double std_skellam = 0; // Standard deviation of the Skellam distribution
  double thsd = 0;        // Chance level of number of excitatory spikes

  for ( long i = 0; i < buf_sz; i++ ){
    if ( B_.in_spikes_.count(tick - i) ){
      // Total weighted net input (positive-negative)
      spike_count_in += B_.in_spikes_[tick - i];
      if (B_.in_spikes_[tick - i]>0){
        // Total weighted positive input
        spike_count_in_pos+=B_.in_spikes_[tick - i];
      }
      else{
        // Total weighted negative input
        spike_count_in_neg+=B_.in_spikes_[tick - i];
      }
    }
  }

  S_.in_rate = 1000.0 * spike_count_in / P_.buffer_size ;

  //std::cout << "-----------" << std::endl;
  //std::cout << origin << "   " << tick << std::endl;
  //std::cout << "Pos_w: " << spike_count_in_pos << std::endl;
  //std::cout << "Neg_w: " << spike_count_in_neg << std::endl;
  //std::cout << "Net_w: " << spike_count_in << std::endl;

  // The difference of two Poisson distributed random variables is distributed according
  // to the Skellam distribution (with coefficients the lamdas of the two Poissons).
  // To compute the difference of the excitatory and the inhibitory channels,
  // one should therefore draw from such a distribution (TODO).
  // For now, I adjust the estimated excitatory-inhibitory difference by the value
  // of this difference that may occur by chance. I estimate this threshold as the
  // standard deviation of a Skellam distribution defined by two equal lamda coefficients,
  // those corresponding to the highest channel (worst case scenario). If the
  // estimated excitatory-inhibitory difference falls lower than this threshold,
  // I consider this difference equal to zero.
  lamda_est   = abs(S_.in_rate);
  std_skellam = sqrt(2*lamda_est);
  thsd        = std_skellam;

  // If the number of spikes is lower than chance, set it to zero
  if ( abs(S_.in_rate)<thsd )
    S_.in_rate = 0;

  // Remove the chance threshold to the number of spikes
  if ( S_.in_rate>0 )
    S_.in_rate = S_.in_rate - thsd;
  if ( S_.in_rate<0 )
    S_.in_rate = S_.in_rate + thsd;

  // Check if neuron is sensitive to positive or negative signals
  if ( (S_.in_rate<0 && P_.pos) || (S_.in_rate>0 && !P_.pos) ){
    S_.in_rate = 0;
  }

  // Multiply by 1000 to translate rate in Hz (buffer size is in milliseconds)
  // Absolute value because spike_count_in could be a negative value (due to
  // negative weights in inhibitory synapses, and negative neurons - i.e. P_pos
  // is false).
  S_.out_rate = std::max(0.0, P_.base_rate + P_.kp * abs(S_.in_rate));

  /*
  std::cout << "std_skl  : " << std_skellam << std::endl;
  std::cout << "thsd     : " << thsd << std::endl;
  std::cout << "In rate  : " << S_.in_rate << std::endl;
  std::cout << "Out rate : " << S_.out_rate << std::endl;
  */

  // Set Poisson lambda respective time resolution
  V_.poisson_dev_.set_lambda( S_.out_rate * time_res * 1e-3 );

  for ( long lag = from ; lag < to ; ++lag ) {
    // Number of output spikes based (i.e. draw from Poisson distribution)
    spike_count_out = V_.poisson_dev_.ldev( rng );

    // Send spike with multiplicity 1 (one could also respect multiplicity)
    if ( spike_count_out > 0 ){
      nest::SpikeEvent se;
      se.set_multiplicity( 1 );
      nest::kernel().event_delivery_manager.send( *this, se, lag );
    }

    // voltage logging
    B_.logger_.record_data(tick+lag);
  }
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void diff_neuron::handle(nest::DataLoggingRequest& e){
  B_.logger_.handle(e);
}


void diff_neuron::handle(nest::SpikeEvent &e){
  assert(e.get_delay_steps() > 0);

  long origin_step       = nest::kernel().simulation_manager.get_slice_origin().get_steps();
  long delivery_step_rel = e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() );
  long tick              = origin_step + delivery_step_rel;

  const double weight       = e.get_weight();
  const double multiplicity = e.get_multiplicity();

  double map_value = 0.0;
  if ( B_.in_spikes_.count(tick) ){
    map_value = B_.in_spikes_[tick];
  }
  B_.in_spikes_[tick] = map_value + weight * multiplicity;

  if ( weight < 0.0 ){ // inhibitory
    get_inh_spikes().add_value(delivery_step_rel, weight * multiplicity );
  }
  if ( weight >= 0.0 ){ // excitatory
    get_exc_spikes().add_value(delivery_step_rel, weight * multiplicity );
  }
}
