/*
*  state_neuron.cpp
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

#include "state_neuron.h"


/* ----------------------------------------------------------------
* Recordables map
* ---------------------------------------------------------------- */
nest::RecordablesMap<state_neuron> state_neuron::recordablesMap_;

namespace nest
{
  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
  template <> void RecordablesMap<state_neuron>::create(){
  // use standard names whereever you can for consistency!

  insert_("in_rate", &state_neuron::get_in_rate);

  insert_("out_rate", &state_neuron::get_out_rate);
  }
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * Note: the implementation is empty. The initialization is of variables
 * is a part of the state_neuron's constructor.
 * ---------------------------------------------------------------- */
state_neuron::Parameters_::Parameters_(){}

state_neuron::State_::State_(){}

/* ----------------------------------------------------------------
* Parameter and state extractions and manipulation functions
* ---------------------------------------------------------------- */

state_neuron::Buffers_::Buffers_(state_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

state_neuron::Buffers_::Buffers_(const Buffers_ &, state_neuron &n):
  logger_(n){
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */
state_neuron::state_neuron():Archiving_Node(), P_(), S_(), B_(*this)
{
  recordablesMap_.create();

  P_.kp = 1; // as real
  P_.pos = true; // as boolean
  P_.num_first = 0.0; // as real
  P_.num_second = 0.0; // as real
  P_.buffer_size = 100*1.0; // as real
  P_.base_rate = 0; // as real
  S_.in_rate = 0; // as real
  S_.out_rate = 0; // as real
}

state_neuron::state_neuron(const state_neuron& __n):
  Archiving_Node(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this){
  P_.kp = __n.P_.kp;
  P_.pos = __n.P_.pos;
  P_.num_first = __n.P_.num_first;
  P_.num_second = __n.P_.num_second;

  P_.buffer_size = __n.P_.buffer_size;
  P_.base_rate = __n.P_.base_rate;

  S_.in_rate = __n.S_.in_rate;
  S_.out_rate = __n.S_.out_rate;

}

state_neuron::~state_neuron(){
}

/* ----------------------------------------------------------------
* Node initialization functions
* ---------------------------------------------------------------- */

void state_neuron::init_state_(const Node& proto){
  const state_neuron& pr = downcast<state_neuron>(proto);
  S_ = pr.S_;
}



void state_neuron::init_buffers_(){

  get_inh_spikes().clear(); //includes resize
  get_exc_spikes().clear(); //includes resize

  B_.logger_.reset(); // includes resize
  Archiving_Node::clear_history();
  if (P_.pos == true){
    std::ofstream fout("variability_pos.txt"); // Initialize output file
  }
  else{
    std::ofstream fout("variability_neg.txt"); // Initialize output file
  }
  

}

void state_neuron::calibrate(){
  B_.logger_.init();
}

/* ----------------------------------------------------------------
* Update and spike handling functions
* ---------------------------------------------------------------- */

/*
 *
 */
void state_neuron::update(nest::Time const & origin,const long from, const long to){

  librandom::RngPtr rng = nest::kernel().rng_manager.get_rng( get_thread() );

  long tick = origin.get_steps();
  double time_res = nest::Time::get_resolution().get_ms();
  long spike_count_out = 0;

  // Update rate
  //if (tick % 500 == 0){ // every 50 ms
  long buf_sz = std::lrint(P_.buffer_size / time_res);
  int num_first = int(P_.num_first);
  int num_second = int(P_.num_second);

  std::map<long, double> spikes_first_buff;
  std::map<long, double> spikes_second_buff;
  std::map<long, double>::iterator it;
  for ( long i = 0; i < buf_sz; i++ ){

    for (it = B_.first_spikes_buffer[tick-i].begin(); it != B_.first_spikes_buffer[tick-i].end(); it++)
    {
      spikes_first_buff[it->first] += double(it->second);
    }

    for (it = B_.second_spikes_buffer[tick-i].begin(); it != B_.second_spikes_buffer[tick-i].end(); it++)
    {
      spikes_second_buff[it->first] += double(it->second);
    }
  }

  // First buffer
  double variability_first;
  double mean_first;
  if (num_first == 0){
    variability_first = 1e6;
    mean_first = 0.0;
  }
  else{
    mean_first = 0;
    for (it = spikes_first_buff.begin(); it != spikes_first_buff.end(); it++)
    {
      mean_first += double(it->second);
    }
    mean_first /= num_first;
    if (mean_first != 0){
      double var_first = 0;
      for (it = spikes_first_buff.begin(); it != spikes_first_buff.end(); it++)
      {
        var_first += (double(it->second) - mean_first) * (double(it->second) - mean_first);
      }
      //var_first = sqrt(var_first/num_first); // standard deviation
      var_first = var_first/num_first; // variance
      variability_first = var_first/mean_first;
    }
    else{
      variability_first = 3.0; // huge value for a CV
    }
  }
  
  

  // Second buffer
  double variability_second;
  double mean_second;
  if (num_second == 0){
    variability_second = 1e6;
    mean_second = 0.0;
  }
  else{
    mean_second = 0;
    for (it = spikes_second_buff.begin(); it != spikes_second_buff.end(); it++)
    {
      mean_second += double(it->second);
    }
    mean_second /= num_second;
    if (mean_second != 0){
      double var_second = 0;
      for (it = spikes_second_buff.begin(); it != spikes_second_buff.end(); it++)
      {
        var_second += (double(it->second) - mean_second) * (double(it->second) - mean_second);
      }
      //var_second = sqrt(var_second/num_second); // standard deviation
      var_second = var_second/num_second; // variance
      variability_second = var_second/mean_second;
    }
    else{
      variability_second = 3.0; // huge value for a CV
    }
  }
  /*
      //std::cout << "Time: " << tick*time_res << std::endl;
      double spikes_first_buff[num_first] = {};
      double spikes_second_buff[num_second] = {};
      std::map<long, double>::iterator it;

      std::ofstream fout;
      fout.open("first_buffer.txt",std::ofstream::app);
      int i = 0;
      for (it = B_.first_spikes_buffer.begin(); it != B_.first_spikes_buffer.end(); it++)
      {
        spikes_first_buff[i] = double(it->second);
        i++;
        fout<< double(it->first) << "," << double(it->second) << std::endl;
      }
      fout.close();
      //std::cout << "Second buffer:" << std::endl;
      fout.open("second_buffer.txt",std::ofstream::app);
      i = 0;
      for (it = B_.second_spikes_buffer.begin(); it != B_.second_spikes_buffer.end(); it++)
      {
           spikes_second_buff[i] = double(it->second);
           i++;
           //std::cout << double(it->second) << ' ';
           fout<< double(it->first) << "," << double(it->second) << std::endl;
      }
      fout.close();
      //std::cout << std::endl;
      //std::cout << (sizeof(spikes_second_buff)/sizeof(*spikes_second_buff)) << std::endl;

      // First buffer
      //std::cout << "Test:" << std::endl;
      double mean_first = 0;
      for(int n = 0; n < num_first; n++ )
      {
        mean_first += spikes_first_buff[n];
        //std::cout << spikes_first_buff[n] << ' ';
      }
      //std::cout << std::endl;
      mean_first /= num_first;
      double variability_first;
      if (mean_first != 0){
        double var_first = 0;
        for(int n = 0; n < num_first; n++ )
        {
          var_first += (spikes_first_buff[n] - mean_first) * (spikes_first_buff[n] - mean_first);
        }
        var_first = sqrt(var_first/num_first); // standard deviation
        variability_first = var_first/mean_first;
      }
      else{
        variability_first = 3.0; // huge value for a CV
      }
      //std::cout << "Mean value first buffer: " << mean << std::endl;
      //std::cout << "Variability first buffer: " << variability << std::endl;

      // Second buffer
      double mean_second = 0;
      for(int n = 0; n < num_second; n++ )
      {
        mean_second += spikes_second_buff[n];
      }
      mean_second /= num_second;
      double variability_second;
      if (mean_second != 0){
        double var_second = 0;
        for(int n = 0; n < num_second; n++ )
        {
          var_second += (spikes_second_buff[n] - mean_second) * (spikes_second_buff[n] - mean_second);
        }
        var_second = sqrt(var_second/num_second);
        variability_second = var_second/mean_second;
      }
      else{
        variability_second = 3.0;
      }
  */ 
  
  // Bayesian integration   
  double total_var = variability_first + variability_second; 
  double out_spikes = variability_first/total_var*mean_second + variability_second/total_var*mean_first;
  out_spikes = std::max(0.0, out_spikes);

  if ((num_first != 0) & (num_second != 0)){
    if (tick % 50 == 0){ // every 5 ms
      std::ofstream fout;
      if (P_.pos == true){
        fout.open("variability_pos.txt",std::ofstream::app);
      }
      else{
        fout.open("variability_neg.txt",std::ofstream::app);
      }
      fout<< "Time (ms): " << tick/10 << "; ";
      fout<< float(variability_first) << ',';
      fout<< float(variability_second) << std::endl;
      fout.close();
    }
    /*
    if (tick % 500 == 0){ // every 50 ms
      std::cout << "First buffer: variability = " << variability_first << " signal = " << mean_first << std::endl;
      std::cout << "Second buffer: variability = " << variability_second << " signal = " << mean_second << std::endl; 
      std::cout << "Bayesian output: " << out_spikes << std::endl;
    }
    */
  }

  // Multiply by 1000 to translate rate in Hz (buffer size is in milliseconds)
  S_.in_rate = 1000.0 * out_spikes / P_.buffer_size ;
  S_.out_rate = P_.base_rate + P_.kp * S_.in_rate;

  // Set Poisson lambda respective time resolution
  V_.poisson_dev_.set_lambda( S_.out_rate * time_res * 1e-3 );
  //std::cout << "In rate: " << S_.in_rate << std::endl;
  
  //} // close update rate

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
void state_neuron::handle(nest::DataLoggingRequest& e){
  B_.logger_.handle(e);
}


void state_neuron::handle(nest::SpikeEvent &e){
  assert(e.get_delay_steps() > 0);

  long origin_step       = nest::kernel().simulation_manager.get_slice_origin().get_steps();
  long delivery_step_rel = e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() );
  long tick              = origin_step + delivery_step_rel;

  const double weight       = std::abs(e.get_weight()); // bayesian integration does not use signs
  //const double multiplicity = e.get_multiplicity();
  const double sender_id    = e.get_sender_gid();

  std::map<long, double>::iterator it; 
  if (e.get_rport() == 1){
    it = B_.first_spikes_buffer[tick].find(sender_id);
    if (it != B_.first_spikes_buffer[tick].end()){
      B_.first_spikes_buffer[tick][sender_id] += weight;
    }
    else{
      B_.first_spikes_buffer[tick][sender_id] = weight;
    }
  }
  else if (e.get_rport() == 2){
    it = B_.second_spikes_buffer[tick].find(sender_id);
    if (it != B_.second_spikes_buffer[tick].end()){
      B_.second_spikes_buffer[tick][sender_id] += weight;
    }
    else{
      B_.second_spikes_buffer[tick][sender_id] = weight;
    }
  }
}
