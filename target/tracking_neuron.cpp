/*
*  tracking_neuron.cpp
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
*  2021-03-01 11:52:50.645009
*/

// C++ includes:
#include <limits>
#include <iostream>
#include <fstream>

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

#include "tracking_neuron.h"


/* ----------------------------------------------------------------
* Recordables map
* ---------------------------------------------------------------- */
nest::RecordablesMap<tracking_neuron> tracking_neuron::recordablesMap_;

namespace nest
{
  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
  template <> void RecordablesMap<tracking_neuron>::create(){
  // use standard names whereever you can for consistency!

  insert_("in_rate", &tracking_neuron::get_in_rate);

  insert_("out_rate", &tracking_neuron::get_out_rate);
  }
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * Note: the implementation is empty. The initialization is of variables
 * is a part of the tracking_neuron's constructor.
 * ---------------------------------------------------------------- */
tracking_neuron::Parameters_::Parameters_(){}

tracking_neuron::State_::State_(){}

/* ----------------------------------------------------------------
* Parameter and state extractions and manipulation functions
* ---------------------------------------------------------------- */

tracking_neuron::Buffers_::Buffers_(tracking_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

tracking_neuron::Buffers_::Buffers_(const Buffers_ &, tracking_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */
tracking_neuron::tracking_neuron():Archiving_Node(), P_(), S_(), B_(*this)
{
  recordablesMap_.create();

  P_.kp = 1; // as real
  P_.pos = true; // as boolean
  P_.repeatable = false; // as boolean
  P_.buffer_size = 100*1.0; // as real
  P_.base_rate = 0; // as real
  //P_.pattern = 0; // as real
  S_.in_rate = 0; // as real
  S_.out_rate = 0; // as real
}

tracking_neuron::tracking_neuron(const tracking_neuron& __n):
  Archiving_Node(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this){
  P_.kp = __n.P_.kp;
  P_.pos = __n.P_.pos;
  P_.repeatable = __n.P_.repeatable;
  P_.buffer_size = __n.P_.buffer_size;
  P_.base_rate = __n.P_.base_rate;
  P_.pattern_file = __n.P_.pattern_file;

  S_.in_rate = __n.S_.in_rate;
  S_.out_rate = __n.S_.out_rate;
}

tracking_neuron::~tracking_neuron(){
}

/* ----------------------------------------------------------------
* Node initialization functions
* ---------------------------------------------------------------- */

void tracking_neuron::init_state_(const Node& proto){
  const tracking_neuron& pr = downcast<tracking_neuron>(proto);
  S_ = pr.S_;
}



void tracking_neuron::init_buffers_(){

  get_inh_spikes().clear(); //includes resize
  get_exc_spikes().clear(); //includes resize

  std::ifstream pattern_file_pt(P_.pattern_file);

  double tmp=0;
  V_.trial_length = 0;
  while(pattern_file_pt >> tmp) {
    V_.pattern.push_back(tmp);
    V_.trial_length++;
  }


  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();

}

void tracking_neuron::calibrate(){
  B_.logger_.init();
}

/* ----------------------------------------------------------------
* Update and spike handling functions
* ---------------------------------------------------------------- */

/*
 *
 */
void tracking_neuron::update(nest::Time const & origin,const long from, const long to){

  librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );

  long tick = origin.get_steps();
  double time_res = nest::Time::get_resolution().get_ms();

  double tmp = 0;
  long spike_count_out = 0;

  // Get signal at the current time stamp
  tmp = V_.pattern[ (int)(tick % V_.trial_length) ];
  //tmp = V_.pattern[ tick ];

  // Check on possible nan values
  if(std::isnan(tmp)){
    tmp = 0;
  }

  // Check if neuron is sensitive to positive or negative signals
  if ( (tmp<0 && P_.pos) || (tmp>=0 && !P_.pos) ){
    tmp = 0;
  }

  if (P_.repeatable == true){
    // init buffer
    if (B_.trial_spikes_.empty() == true)
    {
      nest::Time::ms trial_length_ms(V_.trial_length);
      nest::Time trial_length(trial_length_ms);

      long buffer_size_ = trial_length.get_steps();
      B_.trial_spikes_.resize(buffer_size_);
    }
  }

  S_.out_rate = P_.base_rate + P_.kp * abs(tmp);
  //std::cout << "Out: " << S_.out_rate << std::endl;
  //std::cout << "Tmp: " << tmp << std::endl;
  S_.out_rate = int(S_.out_rate/10)*10.0;

  // TODO: I probably need to scale the signal to make sure neuron does not saturate
  V_.poisson_dev_.set_lambda( S_.out_rate * time_res * 1e-3 );
  //librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );

  for ( long lag = from ; lag < to ; ++lag ) {
    if (P_.repeatable == true){
      nest::delay spike_i = (tick + lag) % V_.trial_length;

      if ( (tick + lag) <= V_.trial_length )
      {
        //librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );
        // Number of output spikes based (i.e. draw from Poisson distribution)
        spike_count_out = V_.poisson_dev_.ldev( rng );        
        B_.trial_spikes_[spike_i] = spike_count_out;
      }
      else
      {
        spike_count_out = B_.trial_spikes_[spike_i];
      }
    }
    else{
      //librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );
      // Number of output spikes based (i.e. draw from Poisson distribution)
      spike_count_out = V_.poisson_dev_.ldev( rng );
    }

    // Send spike with multiplicity 1 (one could also respect multiplicity)
    if ( spike_count_out > 0 ){
      nest::SpikeEvent se;
      se.set_multiplicity( 1 );
      nest::kernel().event_delivery_manager.send( *this, se, lag );
    }

    // voltage logging
    B_.logger_.record_data(origin.get_steps()+lag);
  }

}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void tracking_neuron::handle(nest::DataLoggingRequest& e){
  B_.logger_.handle(e);
}


void tracking_neuron::handle(nest::SpikeEvent &e){
  assert(e.get_delay_steps() > 0);

  const double weight = e.get_weight();
  const double multiplicity = e.get_multiplicity();

  if ( weight < 0.0 ){ // inhibitory
    get_inh_spikes().
        add_value(e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),

                       weight * multiplicity );
  }
  if ( weight >= 0.0 ){ // excitatory
    get_exc_spikes().
        add_value(e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),
                       weight * multiplicity );
  }
}
