/* spec.h -- agent command line specs.
 *
 *   random
 *   heur
 *   net:PATH[:draw_samples]
 *   policy:PATH[:temperature]
 *   mcts:PATH[:dets[:sims[:root_width[:node_width]]]]
 */
#ifndef SPEC_H
#define SPEC_H

#include "agent.h"

/* Parses spec into *a, loading a network if needed.  Exits on error. */
void spec_parse(const char *spec, Agent *a);

#endif
