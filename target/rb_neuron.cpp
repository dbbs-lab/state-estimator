/*
*  rb_neuron.cpp
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

#include "rb_neuron.h"


/* ----------------------------------------------------------------
* Recordables map
* ---------------------------------------------------------------- */
nest::RecordablesMap<rb_neuron> rb_neuron::recordablesMap_;

namespace nest
{
  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
  template <> void RecordablesMap<rb_neuron>::create(){
  // use standard names whereever you can for consistency!

  insert_("in_rate", &rb_neuron::get_in_rate);

  insert_("out_rate", &rb_neuron::get_out_rate);
  }
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * Note: the implementation is empty. The initialization is of variables
 * is a part of the rb_neuron's constructor.
 * ---------------------------------------------------------------- */
rb_neuron::Parameters_::Parameters_(){}

rb_neuron::State_::State_(){}

/* ----------------------------------------------------------------
* Parameter and state extractions and manipulation functions
* ---------------------------------------------------------------- */

rb_neuron::Buffers_::Buffers_(rb_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

rb_neuron::Buffers_::Buffers_(const Buffers_ &, rb_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */
rb_neuron::rb_neuron():Archiving_Node(), P_(), S_(), B_(*this)
{
  recordablesMap_.create();

  P_.kp = 1; // as real
  P_.desired = 0; // as boolean
  P_.buffer_size = 100*1.0; // as real
  P_.base_rate = 0; // as real
  S_.in_rate = 0; // as real
  S_.out_rate = 0; // as real
}

rb_neuron::rb_neuron(const rb_neuron& __n):
  Archiving_Node(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this){
  P_.kp = __n.P_.kp;
  P_.desired = __n.P_.desired;
  P_.buffer_size = __n.P_.buffer_size;
  P_.base_rate = __n.P_.base_rate;

  S_.in_rate = __n.S_.in_rate;
  S_.out_rate = __n.S_.out_rate;

}

rb_neuron::~rb_neuron(){
}

/* ----------------------------------------------------------------
* Node initialization functions
* ---------------------------------------------------------------- */

void rb_neuron::init_state_(const Node& proto){
  const rb_neuron& pr = downcast<rb_neuron>(proto);
  S_ = pr.S_;
}



void rb_neuron::init_buffers_(){

  get_inh_spikes().clear(); //includes resize
  get_exc_spikes().clear(); //includes resize

  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();

}

void rb_neuron::calibrate(){
  B_.logger_.init();
}

/* ----------------------------------------------------------------
* Update and spike handling functions
* ---------------------------------------------------------------- */

/*
 *
 */
void rb_neuron::update(nest::Time const & origin,const long from, const long to){

  librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );

  long tick = origin.get_steps();
  double time_res = nest::Time::get_resolution().get_ms();

  long buf_sz = std::lrint(P_.buffer_size / time_res);
  double spike_count_in = 0;
  double spike_count_out = 0;
  double sdev = 5.0;

  for ( long i = 0; i < buf_sz; i++ ){
    if ( B_.in_spikes_.count(tick - i) ){
      // Total weighted net input (positive-negative)
      spike_count_in += B_.in_spikes_[tick - i];
    }
  }

  S_.in_rate = P_.kp * spike_count_in / P_.buffer_size;
  // Multiply by 1000 to translate rate in Hz (buffer size is in milliseconds)

  //std::cout << "Desired: " << P_.desired << std::endl;
  //std::cout << "In rate: " << S_.in_rate << std::endl;

  S_.out_rate = P_.base_rate + 300.0 * exp(-pow(((P_.desired - S_.in_rate) / sdev), 2 ));

  //std::cout << "Out rate: " << S_.out_rate << std::endl;

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
void rb_neuron::handle(nest::DataLoggingRequest& e){
  B_.logger_.handle(e);
}


void rb_neuron::handle(nest::SpikeEvent &e){
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
}
