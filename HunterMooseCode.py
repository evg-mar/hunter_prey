#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: evgeniy
"""

import numpy as np
import itertools

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


class PlayerBasic(object):
    def __init__(self, label, 
                 current_pos, 
                 moves, 
#                 weights=None,
                 start_limit=None,
                 end_limit=None):
        self.start_pos = current_pos

        self.current_pos = current_pos
        self.current_pos_proba = 1.0
        self.current_step = 0
#        self.plot_moves = {self.current_step: {self.current_pos: 1.0}
#                           }

        self.moves = moves
#        self.weights = weights if weights is not None else \
#                       [1/len(moves)]*len(moves)
#        assert(len(self.moves) == len(self.weights))

        self.start_limit = start_limit
        self.end_limit = end_limit


class Player(PlayerBasic):
    
    def __init__(self, label, 
                 current_pos, 
                 moves, 
                 weights=None,
                 start_limit=None,
                 end_limit=None):
        super(Player, self).__init__(label, 
                                     current_pos, 
                                     moves, 
                                     start_limit, 
                                     end_limit)
        self.plot_moves = {self.current_step: {self.current_pos: 1.0}
                           }


        self.weights = weights if weights is not None else \
                       [1/len(moves)]*len(moves)
        assert(len(self.moves) == len(self.weights))


#    def get_next_position(self, idx):
#        assert(0 <= idx < len(self.moves))
#        next_pos = self.current_pos + self.moves[idx]
#        weight = self.weights[idx]
#        return next_pos, weight
#        
#    def next_move(self, idx):
#        assert(0 <= idx < len(self.moves))
#        self.current_pos, weight = self.get_next_position(idx)
#        self.current_pos_proba *= weight
#        return self.current_pos, self.current_pos_proba
        
    def get_next_moves(self, input_positions, conditional_proba=1.0):
        result = { }
        for position, proba in input_positions.items():
            temp = self.get_next_move(position, proba, conditional_proba)
            result = {k:result.get(k,0.0) + temp.get(k,0.0) for k in set(result)|set(temp)}
        return result
    
    def get_next_move(self, current_pos, proba, conditional_proba=1.0):
        result = { }
        for move,weight in zip(self.moves, self.weights):
            next_pos = current_pos + move
#            print('weight pos: %d'% proba)
#            print('weight: %d' % weight)
            weight_next_pos = proba * (conditional_proba * weight) 
            result[next_pos] = result.get(next_pos, 0.0) + weight_next_pos
        return result

    def next_moves(self):
        result = { }
        current_plot = self.plot_moves[self.current_step]
#        print(current_plot)
        self.current_step += 1
        
        for current_pos, weight_pos in current_plot.items():
            for idx, (move, weight) in enumerate(zip(self.moves, self.weights)):
                next_pos = current_pos + move
#                print('weight pos: %d'% weight_pos)
#                print('weight: %d' % weight)
                weight_next_pos = weight_pos * weight 
                result[next_pos] = result.get(next_pos, 0.0) + weight_next_pos
        
        self.current_step += 1
        self.plot_moves[self.current_step] = result
        

class GameSimulator(object):
    def __init__(self, hunter, prey, die_size=6):
        assert(die_size >= len(hunter.moves) + len(prey.moves))
        self.hunter = hunter
        self.hunter_weight = len(hunter.moves)/die_size
        self.prey = prey
        self.prey_weight = len(prey.moves)/die_size
        self.die_size = die_size
        self.current_step = 0
        self.plot_moves = {
            self.current_step: {(hunter.current_pos, prey.current_pos): 1.0}
            }

        self.hunter_win = { }
        self.prey_win = { }
            
    @property
    def range_board(self):
        return range(self.hunter.start_pos, self.prey.end_limit + 1)

    def next_step(self):
        result = { }
        plot_moves = self.plot_moves[self.current_step]
        if len(plot_moves) == 0:
            print("The simulation has finished!")
            return
    
        for (hunt_pos, prey_pos), proba in plot_moves.items():
            moves = self.hunter.get_next_move(hunt_pos, proba,
                                              self.hunter_weight)
#            print(moves)
            for pos, proba_pos in moves.items():
                proba_new = proba_pos + result.get((pos,prey_pos), 0.0)
                result[pos, prey_pos] = proba_new 
                
            moves = self.prey.get_next_move(prey_pos, proba, self.prey_weight)
#            print(moves)
            for pos, proba_pos in moves.items():
                proba_new = proba_pos + result.get((hunt_pos,pos), 0.0)
                result[hunt_pos, pos] = proba_new

        hunter_win = { }
        prey_win = { }
        for (hunt_pos, prey_pos), proba in list(result.items()):
            
            if hunt_pos >= prey_pos:
                hunter_win[hunt_pos,prey_pos] = proba
                del result[hunt_pos, prey_pos]
            if prey_pos >= self.prey.end_limit:
                prey_win[hunt_pos,prey_pos] = proba
                del result[hunt_pos, prey_pos]

        self.current_step += 1
        if len(hunter_win) > 0:
            self.hunter_win[self.current_step] = hunter_win
        if len(prey_win) > 0:
            self.prey_win[self.current_step] = prey_win
        self.plot_moves[self.current_step] = result
#        if len(result) == 0:
            

            
    def run_simulation(self):
        print("Start simulation...")
        while len(self.plot_moves[self.current_step]) > 0:
            self.next_step()
        
        print("The simulation finished!")
    ###############################################################

    def proba_win_(self, label, player_win_bysteps):
        proba_sum = 0.0
        print("%s win:" % label)
        print('---------')
        for step in range(1, self.current_step+1):
            if step not in player_win_bysteps.keys():
                proba = 0.0
            else:
                player_win = player_win_bysteps[step]
                proba = sum(player_win.values())
            proba_sum +=  proba
            print("Step: %d, %s win proba: %.5f" % (step, label, proba))
        print('---------')
        print("%s win proba sum: %.5f" % (label,proba_sum))

        
    
    def proba_prey_win(self):
        self.proba_win_("Prey", self.prey_win)        
        
    def proba_hunter_win(self):
        self.proba_win_("Hunter", self.hunter_win)
        
    def print_stats(self):
        print("------------------------------------")
        self.proba_hunter_win()
        self.proba_prey_win()
        print("------------------------------------")
    ###############################################################    
        
    def get_proba_final_bystep(self, step):
        assert len(self.plot_moves.get(self.current_step, {})) == 0, \
               "There are intermediate moves. Simulation not finished!"
        assert 0 <= step <= self.current_step, \
               "Last simulation step is %d, number %d was given"  % \
               (self.current_step, step)
        
        hunter_win = self.hunter_win.get(step, {})
        hunter_proba = sum(hunter_win.values())
        
        prey_win   = self.prey_win.get(step, {})
        prey_proba = sum(prey_win.values())
        
        return hunter_proba, prey_proba
        
    def generate_proba_distribution_win(self):
        hunter_proba = { }
        prey_proba = { }
        for step in range(1, self.current_step+1):
            hunt, prey = self.get_proba_final_bystep(step)
            hunter_proba[step] = hunt
            prey_proba[step]   = prey

        return hunter_proba, prey_proba

    #############################################################    
    def get_proba_bystep(self, step, plot_range):
        assert len(self.plot_moves.get(self.current_step, {})) == 0, \
               "There are intermediate moves. Simulation not finished!"
        assert 0 <= step <= self.current_step, \
               "Last simulation step is %d, number %d was given"  % \
               (self.current_step, step)
#        length = plot_range
        start_pos = self.hunter.start_pos
        result_arr = np.zeros((plot_range, plot_range),
                              dtype=float)

        def update_array(dict_holder):
            for (id01,id02), proba in dict_holder.items():
#                result_arr[id01-start_pos, id02-start_pos] += proba
                result_arr[id01, id02] += proba

        
        for st in range(1, step+1):
            update_array(self.hunter_win.get(st, {}))
            update_array(self.prey_win.get(st, {}))
        
        update_array(self.plot_moves[step])

        return result_arr
        
#    def get_proba_detailed_bystep(self, step):
#       arr = np.sum(self.get_proba_bystep(st) for st in range(1, step+1))
#       s = np.sum(np.sum(arr, axis=0))
#       print(s)
#       return arr

def plot_matrix_proba(array, labels, axes, 
                      prey_end=11):
    axes.set_aspect(1)
    axes.grid(True)
    df_cm = pd.DataFrame(array, 
                         index = labels,
                         columns = labels)

    s_obj = sn.heatmap(df_cm, annot=True, ax=axes, linewidths=3)
#    print(s_obj)
    
    axes.plot([0,len(array)], [len(array),0], 
                                    color='red',
                                    label='Hunter limit')
    axes.set_xlabel("Positions of the Prey.", fontsize=15)
    axes.set_ylabel("Positions of the Hunter.", fontsize=15)    
    axes.plot([prey_end, prey_end], [0, len(array)], 
              color='green',
              label='Prey limit' )
    leg = axes.legend(loc='lower left', 
                      title='LEGEND',
                      fontsize=15,
                      frameon=True) #, bbox_to_anchor=(0.5,-0.1))    
    leg.get_frame().set_edgecolor('b')
    plt.setp(leg.get_title(),fontsize='medium')
    leg.set_zorder(100)

    
def plot_matrix_proba_(df, labels, prey_end=11):

    sn.heatmap(df, annot=True, linewidths=3)
    
    plt.plot([0,len(df)], [len(df),0], 
                                    color='red',
                                    label='Hunter limit')
#    plt.axis.set_xlabel("Positions of the Prey.", fontsize=15)
#    plt.axis.set_ylabel("Positions of the Hunter.", fontsize=15)    
    plt.plot([prey_end, prey_end], [0, len(df)], 
                  color='green',
                  label='Prey limit' )
    leg = plt.legend(loc='lower left', 
                      title='LEGEND',
                      fontsize=15,
                      frameon=True) #, bbox_to_anchor=(0.5,-0.1))    
    leg.get_frame().set_edgecolor('b')
    plt.setp(leg.get_title(),fontsize='medium')
 
    
def plot_stats_bystep(game_simulator):
    gs = game_simulator
    
    ##################################
    f2, ax2 = plt.subplots(1)    
    
    hunt, prey = gs.generate_proba_distribution_win()
    
    hunt = np.array(list(hunt.values()))
    prey = np.array(list(prey.values()))

    xx = np.array(range(1, gs.current_step+1))
    
#    print(sum(hunt) + sum(prey))
    
    ax2.bar(xx - 0.5, 
            hunt, 
            label="Hunter win proba for the step",
            width=0.45,
            color='red',
            alpha=0.4)
    
    ax2.bar(xx - 0.5, 
            prey,
            label="Prey win proba for the step",
            bottom=hunt, 
            width=0.45,
            color='green',
            alpha=0.4)

    ax2.bar(xx - 0.5 + 0.46, 
            prey + hunt,
            label="Odds to finish on the step",
            width=0.45,
            color='blue',
            alpha=0.4)

    hunt_norm = hunt/np.sum(hunt)
    hunt_mean = np.sum(hunt_norm*np.arange(1, gs.current_step+1))
    
#    print(hunt_mean)

    prey_norm = prey/np.sum(prey)    
    prey_mean = np.sum(prey_norm*np.arange(1, gs.current_step+1))

#    print(prey_mean)

    ax2.axvline(hunt_mean,
                label="Expected Hunter win: %.3f"%hunt_mean,
                color='red',
                alpha=1,
                linestyle='dashed', 
                linewidth=5)
 
    ax2.axvline(prey_mean, 
                label="Exptected Prey win: %.3f"%prey_mean,
                color='green',
                alpha=1,
                linestyle='dashed', 
                linewidth=5)

    finish_mean = np.sum((prey+hunt)*np.arange(1, gs.current_step+1)) 
    ax2.axvline(finish_mean, 
                label="Expected finish of game: %.3f"%finish_mean,
                color='blue',
                alpha=1,
                linestyle='dashed', 
                linewidth=5)
    
    ax2.set_xlabel("Step number", fontsize=15)
    ax2.set_ylabel("Odds ot win on the step", fontsize=15)
    leg = ax2.legend(loc='upper right', 
                      title='LEGEND',
                      fontsize=15,
                      frameon=True) #, bbox_to_anchor=(0.5,-0.1))    
    leg.get_frame().set_edgecolor('b')
    plt.setp(leg.get_title(),fontsize='medium')

    xticks = ax2.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)

    
    return f2, ax2
    

def plot_stats_sharey_bystep(game_simulator):
    
    gs = game_simulator
    #############################################        
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
     
#    fig.subplots_adjust(vspace=0.5)
#    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    #############################################
    xx = np.array(range(1, gs.current_step+1))
        
    hunt, prey = gs.generate_proba_distribution_win()
    
    hunt = np.array(list(hunt.values()))
    prey = np.array(list(prey.values()))
    
#    print(sum(hunt) + sum(prey))
    
    ax1.bar(xx - 0.5, 
            hunt, 
            label="Hunter win proba for the step",
            width=0.45,
            color='red',
            alpha=0.4)
    
    ax1.bar(xx- 0.5 + 0.46, 
            prey,
            label="Prey win proba for the step",
            width=0.45,
            color='green',
            alpha=0.4)

    hunt_norm = hunt/np.sum(hunt)
    hunt_mean = np.sum(hunt_norm*np.arange(1, gs.current_step+1))
    
#    print(hunt_mean)

    prey_norm = prey/np.sum(prey)    
    prey_mean = np.sum(prey_norm*np.arange(1, gs.current_step+1))

#    print(prey_mean)

    ax1.axvline(hunt_mean,
                label="Expected Hunter win: %.3f"%hunt_mean,
                color='red',
                alpha=1,
                linestyle='dashed', 
                linewidth=5)
 
    ax1.axvline(prey_mean, 
                label="Exptected Prey win: %.3f"%prey_mean,
                color='green',
                alpha=1,
                linestyle='dashed', 
                linewidth=5)

    ax1.set_xlabel("Step number", fontsize=15)
    ax1.set_ylabel("Odds ot win on the step", fontsize=15)

    leg = ax1.legend(loc='upper right', 
                      title='LEGEND',
                      fontsize=15,
                      frameon=True) #, bbox_to_anchor=(0.5,-0.1))    
    leg.get_frame().set_edgecolor('b')
    plt.setp(leg.get_title(),fontsize='medium')


    ############################################
    
    ax2.bar(xx - 0.5, 
            prey + hunt,
            label="Odds to finish on the step",
            width=0.98,
            color='blue',
            alpha=0.4)


    finish_mean = np.sum((prey+hunt)*np.arange(1, gs.current_step+1)) 
    ax2.axvline(finish_mean, 
                label="Expected finish of game: %.3f"%finish_mean,
                color='blue',
                alpha=1,
                linestyle='dashed', 
                linewidth=5)
    
#    plt.xticks(xx[:-1] + 0.5, labels)
#    labels=list(map(str, list( range(1, gs.prey.end_limit+6) )))
    ##################################
    ax2.set_xlabel("Step number", fontsize=15)
    ax2.set_ylabel("Odds ot win on the step", fontsize=15)

    leg = ax2.legend(loc='upper right', 
                      title='LEGEND',
                      fontsize=15,
                      frameon=True) #, bbox_to_anchor=(0.5,-0.1))    
    leg.get_frame().set_edgecolor('b')
    plt.setp(leg.get_title(),fontsize='medium')
    
   
    # Hide first and last ticks - not needed
    xticks = ax1.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)

    xticks = ax2.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    
    
    return fig, (ax1, ax2)    
    
if __name__ == "__main__":
    hunter_moves = [5, 6]
    hunter_weights = None
    [1/2, 1/2]
    
    prey_moves = [1, 2, 3, 4]
    prey_weights = None
    [1/4, 1/4, 1/4, 1/4]
    
    hunter_start = 1
    prey_start = 7
    prey_end = 12
    
    hunter = Player("Hunter", current_pos=hunter_start-1,
                    moves=hunter_moves, 
                    weights=hunter_weights)
    prey = Player("Prey", current_pos=prey_start-1, 
                    moves=prey_moves, 
                    weights=prey_weights,
                    end_limit=prey_end-1)
    
            
    gs = GameSimulator(hunter, prey, die_size=6)    

    gs.run_simulation()
    

    ax2 = plot_stats_bystep(gs)
    
    ax01, ax02 = plot_stats_sharey_bystep(gs)
    
    f, ax1 = plt.subplots(1)
    
    
    plot_range = gs.prey.end_limit + gs.die_size
    labels=list(map(str, list( range(1, plot_range+1) )))
    
    arr3 = gs.get_proba_bystep(gs.current_step, len(labels))
    

    plot_matrix_proba(arr3, 
                      labels=labels,
                      axes=ax1,
                      prey_end=gs.prey.end_limit)
    plt.show()