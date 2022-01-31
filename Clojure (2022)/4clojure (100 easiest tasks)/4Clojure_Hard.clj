;--53. 
;Longest Increasing Sub-Seq
;; Given a vector of integers, find the longest consecutive sub-sequence of increasing numbers. 
;; If two sub-sequences have the same length, use the one that occurs first. 
;; An increasing sub-sequence must have a length of 2 or greater to qualify.
;; (= (__ [1 0 1 2 3 0 4 5]) [0 1 2 3])
;; (= (__ [5 6 1 3 2 7]) [5 6])
;; (= (__ [2 3 3 4 5]) [3 4 5])
;; (= (__ [7 6 5 4]) [])
(fn [s]
   (letfn [(cf [res ss ]              
             (if (not-empty ss)              
               (if (> (first ss) (last (last res)))
                  (cf (update  res (- (count res) 1) #(conj % (first ss))) (rest ss) )                 
                  (cf (conj res (vector(first ss))) (rest ss) ))                
               res))]
     (let [cs (cf [[(first s)]] (rest s) )]       
      ( let [result (first (val(apply max-key key (group-by count cs))))]
       (if (> (count result) 1) result []))))) 

;--73.
;Analyze a Tic-Tac-Toe Board
;; A tic-tac-toe board is represented by a two dimensional vector. 
;; X is represented by :x, O is represented by :o, and empty is represented by :e. 
;; A player wins by placing three Xs or three Os in a horizontal, vertical, or diagonal row. 
;; Write a function which analyzes a tic-tac-toe board and returns 
;; :x if X has won, :o if O has won, and nil if neither player has won.
;; (= nil (__ [[:e :e :e]
;;             [:e :e :e]
;;             [:e :e :e]]))
;; (= :x (__ [[:x :e :o]
;;            [:x :e :e]
;;            [:x :e :o]]))
;; (= :o (__ [[:e :x :e]
;;            [:o :o :o]
;;            [:x :e :x]]))
;; (= nil (__ [[:x :e :o]
;;             [:x :x :e]
;;             [:o :x :o]]))
;; (= :x (__ [[:x :e :e]
;;            [:o :x :e]
;;            [:o :e :x]]))
;; (= :o (__ [[:x :e :o]
;;            [:x :o :e]
;;            [:o :e :x]]))
;; (= nil (__ [[:x :o :x]
;;             [:x :o :x]
;;             [:o :x :o]]))
;not in a "functional style", but I just wanted to try atom
(fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)
        reversed (vec (map #(->> %
                                 reverse
                                 vec) w))]
    (letfn
     [(horizontal [v]
        (if (some (fn [p]
                    (let [fp  (first p)]
                      (when-not (= fp :e)
                        (reset! result fp)
                        (every? #(= % fp) p))))
                  v)
          @result
          nil))
      (diagonal [v]
        (if (and
             (not (= (first (first v)) :e))
             (=
              (first (first v))
              (second (second v))
              (last (last v))))
          (first (first v))
          nil))]
      (or
       (diagonal w)
       (diagonal reversed)
       (horizontal w)
       (horizontal transponed)))))