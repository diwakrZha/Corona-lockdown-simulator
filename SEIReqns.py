#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:12:52 2020

@author: diwaker
"""

def SEIReqns(Y, t, N, sd, beta, gamma, sigma):
    # :param float SD: Social distancing factor
    # :param array x: Time step (days)
    # :param int N: Population
    # :param float beta: The parameter controlling how often a susceptible-infected contact results in a new infection.
    # :param float gamma: The rate an infected recovers and moves into the resistant phase.
    # :param float sigma: The rate at which an exposed person becomes infective.

    S, E, I, R = Y

    dSdt = - sd*beta * S * I / N
    dEdt = sd*beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt