use csv;
use matrixlab::matrix::sparse::{Element, Matrix};
use std::{collections::HashMap, fs, path::Path};

fn main() {
    let mut data = load_amazon();
}

struct Coarsened {
    /* unprocessed graph */
    graph: Matrix<f64>,
    /* transpose of the piecewise constant interpolation matrices for each pass.
     * these can be used to construct a coarse graph at any each level.
     */
    pci: Vec<Matrix<f64>>,
}

#[derive(Debug)]
enum Agg {
    Pair(usize, usize),
    Solo(usize),
}

impl Coarsened {
    /* creates a coarsened graph */
    fn new(graph: Matrix<f64>) -> Self {
        let n = graph.num_rows();
        let mut delta_q = vec![];
        let mut pci = vec![];

        /* calculates row sums */
        let row_sums: Vec<f64> = graph.row_sums();

        /* total of all weights to normalize row sums */
        let total: f64 = row_sums.iter().sum();

        /* generate change in modularity matrix */
        let delta_q_elements: Vec<Element<f64>> = graph
            .elements()
            .for_each(|Element(i, j, v)| {
                let b: f64 = (2.0 / total) * (v - (row_sums[i] * row_sums[j] / total));
                Element::new(i, j, b)
            })
            .collect();
        delta_q.push(n, n, Matrix::new(delta_q_elements));

        delta_q.push(initial_delta_q);
        loop {
            match next(delta_q.pop().unwrap()) {
                Some((delta_q_c, joined)) => {
                    self.joins.push(joined);
                    self.delta_q.push(delta_q_c);
                }
                None => break,
            }
        }
    }

    fn next(delta_q: Matrix<f64>) -> Option<(CsMat<f64>, Vec<Agg>)> {
        let n = delta_q.rows();

        /* p is a vector of vertex indices that a given vertex wants to join with */
        let p: Vec<Option<usize>> = delta_q
            .outer_iterator()
            .enumerate()
            .map(|(i, row_vec)| {
                row_vec
                    .iter()
                    .filter(|(j, _)| i != *j)
                    .fold((None, 0.0), |(col, max), (j, q)| {
                        if *q > 0.0 && *q > max {
                            (Some(j), *q)
                        } else {
                            (col, max)
                        }
                    })
                    .0
            })
            .collect();

        let mut to_join = Vec::with_capacity(n);

        p.iter().enumerate().for_each(|(i, j)| match j {
            None => to_join.push(Agg::Solo(i)),
            Some(j) => match p[*j] {
                None => to_join.push(Agg::Solo(i)),
                Some(p_i) => {
                    if p_i != i {
                        to_join.push(Agg::Solo(i));
                    } else if *j > i {
                        to_join.push(Agg::Pair(i, *j));
                    }
                }
            },
        });

        /* the size of the coarsened graph */
        let nc = to_join.len();

        /* if nothing to merge then complete */
        if nc == n {
            return None;
        }

        /* piecewise constant interpolation/prolongation matrix transpose - the coarsening transformation matrix */
        let mut pci = CsMat::zero((nc, n));

        /* group all the pairs and create the transformation matrix */
        to_join.iter().enumerate().for_each(|(a, agg)| match agg {
            Agg::Solo(j) => pci.insert(a, *j, 1.0),
            Agg::Pair(i, j) => {
                pci.insert(a, *i, 1.0);
                pci.insert(a, *j, 1.0);
            }
        });

        /* calculate the new delta modularity matrix */
        let qp = &pci * &delta_q;
        let delta_q_c = &qp * &pci.transpose_into();

        Some((delta_q_c, to_join))
    }
}

fn load_amazon() -> Matrix<f64> {
    let n = 925872;

    let path = Path::new("./amazon.txt");
    let contents = fs::read_to_string(path).unwrap();

    let mut rdr = csv::Reader::from_reader(contents.as_bytes());

    let mut graph = CsMat::zero((n, n));

    for edge in rdr.records() {
        let edge = edge.unwrap();
        let i = edge[0].parse().unwrap();
        let j = edge[1].parse().unwrap();
        graph.insert(i, j, 1.0);
        graph.insert(j, i, 1.0);
    }

    let word_vec = vec![];
    let joins = vec![];
    let delta_q = vec![];

    Data {
        word_vec,
        graph,
        joins,
        delta_q,
    }
}

/* loads in tweet text into a graph - the vec is the words corresponding to vertices */
fn load() -> (Matrix<f64>, Vec<String>) {
    let mut word_map = HashMap::new();
    let mut word_vec = vec![];

    let path = Path::new("./trump.csv");
    let contents = fs::read_to_string(path).unwrap();

    let mut rdr = csv::Reader::from_reader(contents.as_bytes());
    let mut i: usize = 0;

    /* takes each tweet and puts all the words into its own vec */
    let tweets: Vec<Vec<String>> = rdr
        .records()
        .map(|record| {
            record.unwrap()[0]
                .split_whitespace()
                .map(|word| String::from(word))
                .collect()
        })
        .collect();

    /* iterate over all words and only saves the unique */
    tweets.iter().for_each(|tweet| {
        tweet.iter().for_each(|word| {
            let owned = String::from(word);
            if !word_map.contains_key(&owned) {
                word_map.insert(owned.clone(), i);
                word_vec.push(owned);
                i += 1;
            }
        })
    });

    /* number of vertices in unprocessed graph */
    let n = word_vec.len();
    let mut graph = CsMat::zero((n, n));

    /* iterates over every word and records what words are found in proximity */
    for tweet in tweets.iter() {
        let len = tweet.len();
        for (x, word) in tweet.iter().enumerate() {
            match word_map.get(word) {
                Some(i) => {
                    /* looks at the next 4 words, so words within 8 words get an edge */
                    for y in (x + 1..x + 9).filter(|y| *y < len) {
                        match word_map.get(&tweet[y]) {
                            Some(j) => {
                                let val;
                                /* get the current edge value and increment */
                                match graph.get(*i, *j) {
                                    Some(edge) => val = edge + 1.0,
                                    None => val = 1.0,
                                }
                                graph.insert(*i, *j, val);
                                graph.insert(*j, *i, val);
                            }
                            None => unreachable!(),
                        }
                    }
                }
                None => unreachable!(),
            }
        }
    }

    let joins = vec![];
    let delta_q = vec![];

    Data {
        word_vec,
        graph,
        joins,
        delta_q,
    }
}
