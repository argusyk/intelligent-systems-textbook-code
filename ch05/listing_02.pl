% факти (вимірювання та довідкові дані)
apu(apu123).          apu(apu456).
egt(apu123, 705).     egt(apu456, 610).
start_time(apu123, 52). start_time(apu456, 33).

% правила
overheat(X)     :- apu(X), egt(X, T), T > 680.
slow_start(X)   :- apu(X), start_time(X, S), S > 45.
dirty_nozzles(X) :- overheat(X), slow_start(X).

% запит у консолі SWI-Prolog:
% ?- dirty_nozzles(Which).
% Which = apu123.
