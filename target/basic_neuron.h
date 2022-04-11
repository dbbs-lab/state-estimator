
/*
*  basic_neuron.h
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
*  2021-03-01 11:52:50.555794
*/
#ifndef BASIC_NEURON
#define BASIC_NEURON

#include "config.h"

// Includes from librandom:
#include "poisson_randomdev.h"

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"


// Includes from sli:
#include "dictdatum.h"

/* BeginDocumentation
  Name: basic_neuron.

  Description:


  Parameters:
  The following parameters can be set in the status dictionary.


  Dynamic state variables:


  Initial values:


  References: Empty

  Sends: nest::SpikeEvent

  Receives: Spike,  DataLoggingRequest
*/
class basic_neuron : public nest::Archiving_Node{
public:
  /**
  * The constructor is only used to create the model prototype in the model manager.
  */
  basic_neuron();

  /**
  * The copy constructor is used to create model copies and instances of the model.
  * @node The copy constructor needs to initialize the parameters and the state.
  *       Initialization of buffers and interal variables is deferred to
  *       @c init_buffers_() and @c calibrate().
  */
  basic_neuron(const basic_neuron &);

  /**
  * Releases resources.
  */
  ~basic_neuron();

  /**
   * Import sets of overloaded virtual functions.
   * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
   * Hiding
   */
  using nest::Node::handles_test_event;
  using nest::Node::handle;

  /**
  * Used to validate that we can send nest::SpikeEvent to desired target:port.
  */
  nest::port send_test_event(nest::Node& target, nest::rport receptor_type, nest::synindex, bool);

  /**
  * @defgroup mynest_handle Functions handling incoming events.
  * We tell nest that we can handle incoming events of various types by
  * defining @c handle() and @c connect_sender() for the given event.
  * @{
  */
  void handle(nest::SpikeEvent &);        //! accept spikes
  void handle(nest::DataLoggingRequest &);//! allow recording with multimeter

  nest::port handles_test_event(nest::SpikeEvent&, nest::port);
  nest::port handles_test_event(nest::DataLoggingRequest&, nest::port);
  /** @} */

  // SLI communication functions:
  void get_status(DictionaryDatum &) const;
  void set_status(const DictionaryDatum &);

private:
  //! Reset parameters and state of neuron.

  //! Reset state of neuron.
  void init_state_(const Node& proto);

  //! Reset internal buffers of neuron.
  void init_buffers_();

  //! Initialize auxiliary quantities, leave parameters and state untouched.
  void calibrate();

  //! Take neuron through given time interval
  void update(nest::Time const &, const long, const long);

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap<basic_neuron>;
  friend class nest::UniversalDataLogger<basic_neuron>;

  /**
  * Free parameters of the neuron.
  *
  *
  *
  * These are the parameters that can be set by the user through @c `node.set()`.
  * They are initialized from the model prototype when the node is created.
  * Parameters do not change during calls to @c update() and are not reset by
  * @c ResetNetwork.
  *
  * @note Parameters_ need neither copy constructor nor @c operator=(), since
  *       all its members are copied properly by the default copy constructor
  *       and assignment operator. Important:
  *       - If Parameters_ contained @c Time members, you need to define the
  *         assignment operator to recalibrate all members of type @c Time . You
  *         may also want to define the assignment operator.
  *       - If Parameters_ contained members that cannot copy themselves, such
  *         as C-style arrays, you need to define the copy constructor and
  *         assignment operator to copy those members.
  */
  struct Parameters_{

    double kp;
    bool pos;
    double buffer_size;
    double base_rate;

    /** Initialize parameters to their default values. */
    Parameters_();
  };

  /**
  * Dynamic state of the neuron.
  *
  *
  *
  * These are the state variables that are advanced in time by calls to
  * @c update(). In many models, some or all of them can be set by the user
  * through @c `node.set()`. The state variables are initialized from the model
  * prototype when the node is created. State variables are reset by @c ResetNetwork.
  *
  * @note State_ need neither copy constructor nor @c operator=(), since
  *       all its members are copied properly by the default copy constructor
  *       and assignment operator. Important:
  *       - If State_ contained @c Time members, you need to define the
  *         assignment operator to recalibrate all members of type @c Time . You
  *         may also want to define the assignment operator.
  *       - If State_ contained members that cannot copy themselves, such
  *         as C-style arrays, you need to define the copy constructor and
  *         assignment operator to copy those members.
  */
  struct State_{


    double in_rate;

    double out_rate;
        State_();
  };

  /**
  * Internal variables of the neuron.
  *
  *
  *
  * These variables must be initialized by @c calibrate, which is called before
  * the first call to @c update() upon each call to @c Simulate.
  * @node Variables_ needs neither constructor, copy constructor or assignment operator,
  *       since it is initialized by @c calibrate(). If Variables_ has members that
  *       cannot destroy themselves, Variables_ will need a destructor.
  */
  struct Variables_ {
    librandom::PoissonRandomDev poisson_dev_; //!< Random deviate generator
  };

  /**
    * Buffers of the neuron.
    * Ususally buffers for incoming spikes and data logged for analog recorders.
    * Buffers must be initialized by @c init_buffers_(), which is called before
    * @c calibrate() on the first call to @c Simulate after the start of NEST,
    * ResetKernel or ResetNetwork.
    * @node Buffers_ needs neither constructor, copy constructor or assignment operator,
    *       since it is initialized by @c init_nodes_(). If Buffers_ has members that
    *       cannot destroy themselves, Buffers_ will need a destructor.
    */
  struct Buffers_ {
    Buffers_(basic_neuron &);
    Buffers_(const Buffers_ &, basic_neuron &);

    /** Logger for all analog data */
    nest::UniversalDataLogger<basic_neuron> logger_;

    inline nest::RingBuffer& get_inh_spikes() {return inh_spikes;}
    //!< Buffer incoming pAs through delay, as sum
    nest::RingBuffer inh_spikes;
    double inh_spikes_grid_sum_;

    inline nest::RingBuffer& get_exc_spikes() {return exc_spikes;}
    //!< Buffer incoming pAs through delay, as sum
    nest::RingBuffer exc_spikes;
    double exc_spikes_grid_sum_;

    std::map<long, double> in_spikes_;

    };
  inline double get_in_rate() const {
    return S_.in_rate;
  }
  inline void set_in_rate(const double __v) {
    S_.in_rate = __v;
  }

  inline double get_out_rate() const {
    return S_.out_rate;
  }
  inline void set_out_rate(const double __v) {
    S_.out_rate = __v;
  }

  inline double get_kp() const {
    return P_.kp;
  }
  inline void set_kp(const double __v) {
    P_.kp = __v;
  }

  inline bool get_pos() const {
    return P_.pos;
  }
  inline void set_pos(const bool __v) {
    P_.pos = __v;
  }

  inline double get_buffer_size() const {
    return P_.buffer_size;
  }
  inline void set_buffer_size(const double __v) {
    P_.buffer_size = __v;
  }

  inline double get_base_rate() const {
    return P_.base_rate;
  }
  inline void set_base_rate(const double __v) {
    P_.base_rate = __v;
  }



  inline nest::RingBuffer& get_inh_spikes() {return B_.get_inh_spikes();};

  inline nest::RingBuffer& get_exc_spikes() {return B_.get_exc_spikes();};


  // Generate function header

  /**
  * @defgroup pif_members Member variables of neuron model.
  * Each model neuron should have precisely the following four data members,
  * which are one instance each of the parameters, state, buffers and variables
  * structures. Experience indicates that the state and variables member should
  * be next to each other to achieve good efficiency (caching).
  * @note Devices require one additional data member, an instance of the @c Device
  *       child class they belong to.
  * @{
  */
  Parameters_ P_;  //!< Free parameters.
  State_      S_;  //!< Dynamic state.
  Variables_  V_;  //!< Internal Variables
  Buffers_    B_;  //!< Buffers.

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap<basic_neuron> recordablesMap_;



/** @} */
}; /* neuron basic_neuron */

inline nest::port basic_neuron::send_test_event(
    nest::Node& target, nest::rport receptor_type, nest::synindex, bool){
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c nest::SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender(*this);
  return target.handles_test_event(e, receptor_type);
}

inline nest::port basic_neuron::handles_test_event(nest::SpikeEvent&, nest::port receptor_type){

    // You should usually not change the code in this function.
    // It confirms to the connection management system that we are able
    // to handle @c SpikeEvent on port 0. You need to extend the function
    // if you want to differentiate between input ports.
    if (receptor_type != 0)
      throw nest::UnknownReceptorType(receptor_type, get_name());
    return 0;
}



inline nest::port basic_neuron::handles_test_event(
    nest::DataLoggingRequest& dlr, nest::port receptor_type){
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if (receptor_type != 0)
  throw nest::UnknownReceptorType(receptor_type, get_name());

  return B_.logger_.connect_logging_device(dlr, recordablesMap_);
}

// TODO call get_status on used or internal components
inline void basic_neuron::get_status(DictionaryDatum &__d) const{
  def<double>(__d, "kp", get_kp());

  def<bool>(__d, "pos", get_pos());

  def<double>(__d, "buffer_size", get_buffer_size());

  def<double>(__d, "base_rate", get_base_rate());

  def<double>(__d, "in_rate", get_in_rate());

  def<double>(__d, "out_rate", get_out_rate());
    Archiving_Node::get_status( __d );



  (*__d)[nest::names::recordables] = recordablesMap_.get_list();


}

inline void basic_neuron::set_status(const DictionaryDatum &__d){

  double tmp_kp = get_kp();
  updateValue<double>(__d, "kp", tmp_kp);


  bool tmp_pos = get_pos();
  updateValue<bool>(__d, "pos", tmp_pos);


  double tmp_buffer_size = get_buffer_size();
  updateValue<double>(__d, "buffer_size", tmp_buffer_size);


  double tmp_base_rate = get_base_rate();
  updateValue<double>(__d, "base_rate", tmp_base_rate);


  double tmp_in_rate = get_in_rate();
  updateValue<double>(__d, "in_rate", tmp_in_rate);



  double tmp_out_rate = get_out_rate();
  updateValue<double>(__d, "out_rate", tmp_out_rate);

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status(__d);

  // if we get here, temporaries contain consistent set of properties


  set_kp(tmp_kp);



  set_pos(tmp_pos);



  set_buffer_size(tmp_buffer_size);



  set_base_rate(tmp_base_rate);



  set_in_rate(tmp_in_rate);



  set_out_rate(tmp_out_rate);



};

#endif /* #ifndef BASIC_NEURON */
