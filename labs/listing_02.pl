flight(kyiv, warsaw).   flight(warsaw, paris).
flight(paris, lisbon).  flight(kyiv, vienna).
flight(vienna, paris).

reachable(X, Y) :- flight(X, Y).
reachable(X, Y) :- flight(X, Z), reachable(Z, Y).

route(X, Y, [X, Y]) :- flight(X, Y).
route(X, Y, [X | R]) :- flight(X, Z), route(Z, Y, R).

% ?- reachable(kyiv, lisbon).        true
% ?- route(kyiv, lisbon, R).         R = [kyiv, warsaw, paris, lisbon] ;
%                                    R = [kyiv, vienna, paris, lisbon].
