from shyft import api


class PyPTGSK(object):

    def __init__(self, initial_state, parameter):
        self.pt = api.PriestleyTaylor(parameter.pt.albedo(), parameter.pt.alpha()).potential_evapotranspiration
        self.ae = api.actual_evapotranspiration
        self.gs = api.GS()
        self.k = api.KirchnerMethod(parameter.kirchner)
        self.state = initial_state
        self.parameter = parameter
        self.response = api.PTGSKResp()

    def step(self, t0, t1, temperature, precipitation, rel_hum, radiation, wind_speed, params=None):
        parameter = self.parameter
        pot_evap = self.pt(temperature, radiation, rel_hum)*(t1 - t0)
        self.gs.step(self.state.gs, self.response.gs, t0, t1 - t0, parameter.gs, temperature, radiation, precipitation, wind_speed, rel_hum)
        actual_evap = self.ae(self.state.kirchner.q, pot_evap, parameter.ae.ae_scale_factor(), self.response.gs.sca, t1 - t0)
        self.response.ae.set_ae(actual_evap)
        #q_avg = 0.0
        q, q_avg = self.k.step(t0, t1, self.state.kirchner.q, self.response.gs.outflow, actual_evap)
        self.state.kirchner.set_q(q)
        self.response.kirchner.set_q_avg(q_avg)
        r_frac = 0.1 # no.. parameter.grid.reservoir_fraction()
        discharge = self.response.gs.outflow*r_frac + (1.0 - r_frac)*q_avg
        self.response.total_discharge = discharge
        return q_avg

def main():
    pt_params = api.PriestleyTaylorParameter()
    gs_params = api.GammaSnowParameter()
    ae_params = api.ActualEvapotranspirationParameter()
    k_params = api.KirchnerParameter()
    g_params = api.ScalarPrecipitationCorrectionParameter()

    ptgsk_parameter = api.PTGSKParam(pt_params, gs_params, ae_params, k_params,g_params)

    initial_state = api.PTGSKStat()
    initial_state.kirchner.q = 4.0

    model = PyPTGSK(initial_state, ptgsk_parameter)
    qs = []
    for i in xrange(100):
        if i < 50:
            temperature = 10.0
            precip = 5.0
        else:
            temperature = 2.0
            precip = 0.0
        tmp = model.step(0, 3600, temperature, precip, 0.7, 400, 2.0)
        qs.append(tmp)
    #plot the result:
    from matplotlib import pylab
    pylab.plot(qs)
    pylab.show()


if __name__ == "__main__":
    main()


