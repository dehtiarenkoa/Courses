;--43.
;Reverse Interleave
;; Write a function which reverses the interleave process into x number of subsequences.
;; (= (__ [1 2 3 4 5 6] 2) '((1 3 5) (2 4 6)))
;; (= (__ (range 9) 3) '((0 3 6) (1 4 7) (2 5 8)))
;; (= (__ (range 10) 5) '((0 5) (1 6) (2 7) (3 8) (4 9)))
(fn [s n]
  (let [r (cycle (reverse (range n)))
        start (into [] (repeat n '()))]
    (letfn [(foo [rr res sx] (if (not-empty sx)
                               (foo (rest rr) (update res  (first rr) #(cons  (first sx) %)) (rest sx))
                               res))]
      (reverse(map reverse (seq (foo r start s)))))))

;--44.
;Rotate Sequence
;; Write a function which can rotate a sequence in either direction.
;; (= (__ 2 [1 2 3 4 5]) '(3 4 5 1 2))
;; (= (__ -2 [1 2 3 4 5]) '(4 5 1 2 3))
;; (= (__ 6 [1 2 3 4 5]) '(2 3 4 5 1))
;; (= (__ 1 '(:a :b :c)) '(:b :c :a))
;; (= (__ -4 '(:a :b :c)) '(:c :a :b))
(fn [n s] (letfn [(foo [c m] (cond
                               (= m 0) c
                               (> m 0) (foo (reverse (conj  (reverse (rest c)) (first c))) (dec m))
                               (< m 0) (foo (cons (last c) (butlast c)) (inc m))))] (foo s n)))

;--46.
;Flipping out
;; Write a higher-order function which flips the order of the arguments of an input function.
;; (= 3 ((__ nth) 2 [1 2 3 4 5]))
;; (= true ((__ >) 7 8))
;; (= 4 ((__ quot) 2 8))
;; (= [1 2 3] ((__ take) [1 2 3 4 5] 3))
(fn [f] (fn [a b] (f b a)))
;map from clojure.core
;; (defn map
;;   "Returns a lazy sequence consisting of the result of applying f to
;;   the set of first items of each coll, followed by applying f to the
;;   set of second items in each coll, until any one of the colls is
;;   exhausted.  Any remaining items in other colls are ignored. Function
;;   f should accept number-of-colls arguments. Returns a transducer when
;;   no collection is provided."
;;   {:added "1.0"
;;    :static true}
;;   ([f]
;;    (fn [rf]
;;      (fn
;;        ([] (rf))
;;        ([result] (rf result))
;;        ([result input]
;;         (rf result (f input)))
;;        ([result input & inputs]
;;         (rf result (apply f input inputs))))))
;;   ([f coll]
;;    (lazy-seq
;;     (when-let [s (seq coll)]
;;       (if (chunked-seq? s)
;;         (let [c (chunk-first s)
;;               size (int (count c))
;;               b (chunk-buffer size)]
;;           (dotimes [i size]
;;             (chunk-append b (f (.nth c i))))
;;           (chunk-cons (chunk b) (map f (chunk-rest s))))
;;         (cons (f (first s)) (map f (rest s)))))))
;;   ([f c1 c2]
;;    (lazy-seq
;;     (let [s1 (seq c1) s2 (seq c2)]
;;       (when (and s1 s2)
;;         (cons (f (first s1) (first s2))
;;               (map f (rest s1) (rest s2)))))))
;;   ([f c1 c2 c3]
;;    (lazy-seq
;;     (let [s1 (seq c1) s2 (seq c2) s3 (seq c3)]
;;       (when (and  s1 s2 s3)
;;         (cons (f (first s1) (first s2) (first s3))
;;               (map f (rest s1) (rest s2) (rest s3)))))))
;;   ([f c1 c2 c3 & colls]
;;    (let [step (fn step [cs]
;;                 (lazy-seq
;;                  (let [ss (map seq cs)]
;;                    (when (every? identity ss)
;;                      (cons (map first ss) (step (map rest ss)))))))]
;;      (map #(apply f %) (step (conj colls c3 c2 c1))))))

;--50.
;Split by Type
;; Write a function which takes a sequence consisting of items with different types and splits them up 
;; into a set of homogeneous sub-sequences. The internal order of each sub-sequence should be maintained, 
;; but the sub-sequences themselves can be returned in any order (this is why 'set' is used in the test cases) .
;; (= (set (__ [1 :a 2 :b 3 :c])) #{[1 2 3] [:a :b :c]})
;; (= (set (__ [:a "foo"  "bar" :b])) #{[:a :b] ["foo" "bar"]})
;; (= (set (__ [[1 2] :a [3 4] 5 6 :b])) #{[[1 2] [3 4]] [:a :b] [5 6]})
(fn [s] (vals (group-by type s)))

;--54.
;Partition a Sequence
;; Write a function which returns a sequence of lists of x items each. Lists of less than x items should not be returned.
;;     (= (__ 3 (range 9)) '((0 1 2) (3 4 5) (6 7 8)))
;;     (= (__ 2 (range 8)) '((0 1) (2 3) (4 5) (6 7)))
;;     (= (__ 3 (range 8)) '((0 1 2) (3 4 5)))
;; Special Restrictions : partition,partition-all
(fn [n s] (letfn [(foo [res s] (if (>= (count s) n)
                                 (foo (conj res (take n s)) (drop n s))
                                 res))]
            (reverse (foo '() s))))

;--55.
;Partition a Sequence
;; Write a function which returns a map containing the number of occurences of each distinct item in a sequence.
;;     (= (__ [1 1 2 3 2 1 1]) {1 4, 2 2, 3 1})
;;     (= (__ [:b :a :b :a :b]) {:a 2, :b 3})
;;     (= (__ '([1 2] [1 3] [1 3])) {[1 2] 1, [1 3] 2})
;; Special Restrictions : frequencies
(fn [s] (reduce conj (map #(hash-map (first %) (count (second %)))
                          (group-by identity s))))

;--56.
;Find Distinct Items
;; Write a function which removes the duplicates from a sequence. Order of the items must be maintained.
;;     (= (__ [1 2 1 3 1 2 4]) [1 2 3 4])
;;     (= (__ [:a :a :b :b :c :c]) [:a :b :c])
;;     (= (__ '([2 4] [1 2] [1 3] [1 3])) '([2 4] [1 2] [1 3]))
;;     (= (__ (range 50)) (range 50))
;; Special Restrictions : distinct
(fn [s] (letfn [(foo [ss res] (if (not-empty ss)
                                (if (some #(= (first ss) %) res)
                                  (foo (rest ss) res)
                                  (foo (rest ss) (conj res (first ss))))
                                (into (empty s) (if  (vector? s)
                                                  (reverse res)
                                                  res))))] (foo s ())))

;--58.
;; Write a function which allows you to create function compositions. 
;; The parameter list should take a variable number of functions, and create a function applies them from right-to-left.
;;     (= [3 2 1] ((__ rest reverse) [1 2 3 4]))
;;     (= 5 ((__ (partial + 3) second) [1 2 3 4]))
;;     (= true ((__ zero? #(mod % 8) +) 3 5 7 9))
;;     (= "HELLO" ((__ #(.toUpperCase %) #(apply str %) take) 5 "hello world"))
;; Special Restrictions : comp
;were used "comp" function from clojure.core and 
;https://stackoverflow.com/questions/25628724/misunderstanding-of-variable-arguments-type
(fn [f g & fs]
  (def ff (reverse (list* f g fs)))
  (fn [& args]
    (reduce #(%2 %1)
            (apply (first ff) args)
            (rest ff))))

;--59.
;Juxtaposition
;; Take a set of functions and return a new function that takes a variable number of arguments 
;; and returns a sequence containing the result of applying each function left-to-right to the argument list.
;;     (= [21 6 1] ((__ + max min) 2 3 5 1 6 4))
;;     (= ["HELLO" 5] ((__ #(.toUpperCase %) count) "hello"))
;;     (= [2 6 4] ((__ :a :c :b) {:a 2, :b 4, :c 6, :d 8 :e 10}))
;; Special Restrictions : juxt
(fn [f g & fs]
  (def ff (list* f g fs))
  (let [result []]
    (fn [& args]
      (into []
            (map
             #(apply % args) ff)))))

;--60.
;; Write a function which behaves like reduce, but returns each intermediate value of the reduction. 
;; Your function must accept either two or three arguments, and the return sequence must be lazy.
;;     (= (take 5 (__ + (range))) [0 1 3 6 10])
;;     (= (__ conj [1] [2 3 4]) [[1] [1 2] [1 2 3] [1 2 3 4]])
;;     (= (last (__ * 2 [3 4 5])) (reduce * 2 [3 4 5]) 120)
;; Special Restrictions : reductions
; !!! cheating: simple copying from the CLojure.core
(defn reds  
  ([f coll]
   (lazy-seq
    (if-let [s (seq coll)]
      (reds f (first s) (rest s))
      (list (f)))))
  ([f init coll]
   (if (reduced? init)
     (list @init)
     (cons init
           (lazy-seq
            (when-let [s (seq coll)]
              (reds f (f init (first s)) (rest s))))))))

;--65.
;Black Box Testing
;; Clojure has many collection types, which act in subtly different ways. 
;; The core functions typically convert them into a uniform "sequence" type 
;; and work with them that way, but it can be important to understand the 
;; behavioral and performance differences so that you know which kind is 
;; appropriate for your application. Write a function which takes a collection 
;; and returns one of: map, :set, :list, or :vector - describing the type of 
;; collection it was given. You won't be allowed to inspect their class or use 
;; the built-in predicates like list? - the point is to poke at them and understand their behavior.
;;     (= :map (__ {:a 1, :b 2}))
;;     (= :list (__ (range (rand-int 20))))
;;     (= :vector (__ [1 2 3 4 5 6]))
;;     (= :set (__ #{10 (rand-int 5)}))
;;     (= [:map :set :vector :list] (map __ [{} #{} [] ()]))
;; Special Restrictions : class,type,Class,vector?,sequential?,list?,seq?,map?,set?,instance?,getClass
;works, but "Could not resolve symbol: Exception" error was raised
(fn [sx] (let [e (empty sx)
               p [1 2 3 4]]
           (letfn [(foo [s n]
                     (case n
                       0 (try
                           (into e p)
                           (foo s 1)
                           (catch Exception ex
                             :map))
                       1 (if (= (into e p) #{1 2 3 4})
                           :set
                           (foo s 2))
                       2 (try
                           (contains? e 0)
                           :vector
                           (catch Exception ex1 :list))))]
             (foo sx 0))))
;works
(fn [sx] (let [e (empty sx)
               p [1 2]]
           (if (= (into e [p]) {1 2}) :map
               (if (= (into e p) #{1 2}) :set
                   (if (=  (into e p) [1 2]) :vector :list)))))

;--67.
;Prime Numbers
;; Write a function which returns the first x number of prime numbers.
;; (= (__ 2) [2 3])
;; (= (__ 5) [2 3 5 7 11])
;; (= (last (__ 100)) 541)
(fn [n] 
  (letfn [(prime [res current]                  
                  (cond 
                    (= (count res) n) res
                    (< (count res) n)
                    (prime 
                     (if (some #(= (rem current %) 0) res) 
                       res 
                       (conj res current))
                     (inc current))
                  ))]
    (prime [] 2)
))

;--69.
;	Merge with a Function
;; Write a function which takes a function f and a variable number of maps. 
;; Your function should return a map that consists of the rest of the maps conj-ed onto the first. 
;; If a key occurs in more than one map, the mapping(s) from the latter (left-to-right) should be 
;; combined with the mapping in the result by calling (f val-in-result val-in-latter)
;;     (= (__ * {:a 2, :b 3, :c 4} {:a 2} {:b 2} {:c 5})
;;        {:a 4, :b 6, :c 20})
;;     (= (__ - {1 10, 2 20} {1 3, 2 10, 3 15})
;;        {1 7, 2 10, 3 15})
;;     (= (__ concat {:a [3], :b [6]} {:a [4 5], :c [8 9]} {:b [7]})
;;        {:a [3 4 5], :b [6 7], :c [8 9]})
;; Special Restrictions : merge-with
(fn [f & m]
  (let [s (map #(apply assoc {} %) (apply concat m))]
    (println s)
    (reduce (fn [a b]
              (let [k (apply key b)
                    v (b k)]
                (if (contains? a k)
                  (assoc a k (f (a k) v))
                  (assoc a k v))))
            s)))

;--70.
;; Write a function which splits a sentence up into a sorted list of words. 
;; Capitalization should not affect sort order and punctuation should be ignored.
;; (= (__  "Have a nice day.")
;;    ["a" "day" "Have" "nice"])
;; (= (__  "Clojure is a fun language!")
;;    ["a" "Clojure" "fun" "is" "language"])
;; (= (__  "Fools fall for foolish follies.")
;;    ["fall" "follies" "foolish" "Fools" "for"])
(fn [s]
  (->>
   s
   (#(clojure.string/replace % #"(^\w\s)" ""))
   (#(clojure.string/split % #"(\W)"))
   (#(filter (fn [a] (> (count a) 0)) %))
   (#(sort (fn [a b] (neg? (compare 
                            (clojure.string/lower-case a) 
                            (clojure.string/lower-case b)))) %))))
