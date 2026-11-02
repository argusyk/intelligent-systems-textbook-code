(deftemplate param    (slot name) (slot value))
(deftemplate hyp      (slot name) (slot cf (default 0.0)))
(deftemplate evidence (slot hyp) (slot rule) (slot cf))

(deffunction combine (?a ?b)                       ; формула уточнення Шортліффа
  (if (and (>= ?a 0) (>= ?b 0)) then (+ ?a (* ?b (- 1 ?a)))
   else (if (and (< ?a 0) (< ?b 0)) then (+ ?a (* ?b (+ 1 ?a)))
         else (/ (+ ?a ?b) (- 1 (min (abs ?a) (abs ?b)))))))

(deffacts initial
  (hyp (name H1-nozzles)) (hyp (name H2-regulator))
  (param (name egt) (value 705)) (param (name start) (value 52))
  (param (name fuel) (value 155)))

(defrule R5 "перегрів і затяжний запуск -> форсунки"
  (param (name egt)   (value ?e&:(> ?e 680)))
  (param (name start) (value ?s&:(> ?s 45)))
  => (assert (evidence (hyp H1-nozzles) (rule R5) (cf 0.7))))

(defrule R6 "перегрів і висока витрата -> регулятор"
  (param (name egt)  (value ?e&:(> ?e 680)))
  (param (name fuel) (value ?g&:(> ?g 150)))
  => (assert (evidence (hyp H2-regulator) (rule R6) (cf 0.6))))

(defrule R12 "висока витрата -> форсунки"
  (param (name fuel) (value ?g&:(> ?g 150)))
  => (assert (evidence (hyp H1-nozzles) (rule R12) (cf 0.5))))

(defrule combine-evidence
  ?ev <- (evidence (hyp ?h) (rule ?r) (cf ?c))
  ?hy <- (hyp (name ?h) (cf ?old))
  => (retract ?ev)
     (bind ?new (combine ?old ?c))
     (printout t "  " ?r ": " ?h " " ?old " -> " ?new crlf)
     (modify ?hy (cf ?new)))

(defrule report (declare (salience -10))
  (hyp (name ?h) (cf ?c&:(>= ?c 0.2)))
  => (printout t "ГІПОТЕЗА " ?h " CF = " ?c crlf))
