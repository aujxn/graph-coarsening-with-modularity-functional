use csv;
use sprs::CsMat;
use std::{collections::HashMap, fs, path::Path};

fn main() {
    let mut data = Data::load();
    for _ in 0..15 {
        data.coarsen();
    }

    data.print_clusters();
}

struct Data {
    /* the vertices of the unprocessed graph */
    word_vec: Vec<String>,
    /* index is number of passes through the coarsening algo */
    partitions: Vec<Partition>,
}

struct Partition {
    /* first index is location on original graph and second is the group of vertices */
    aggregates: Vec<Vec<usize>>,
    /* CSR graph of the undirected edges with weights */
    graph: CsMat<usize>,
    /* change in modularity matrix resulting from modularity functional */
    delta_q: CsMat<f64>,
}

impl Partition {
    /* row sums of the coarsened graph */
    fn row_sums(&self) -> Vec<usize> {
        self.graph
            .outer_iterator()
            .map(|row_vec| row_vec.iter().fold(0, |acc, (_, val)| acc + val))
            .collect()
    }
}

impl Data {
    fn print_clusters(&mut self) {
        let last = self.partitions.len() - 1;

        self.partitions[last].aggregates.iter().for_each(|agg| {
            if agg.len() > 1 {
                println!();
                agg.iter().for_each(|i| print!("{:?}, ", self.word_vec[*i]));
            }
        })
    }

    /* performs one level of coarsening on the graph */
    fn coarsen(&mut self) {
        let level = self.partitions.len() - 1;
        let n = self.partitions[level].aggregates.len();

        /* p is a vector of vertex indices that a given vertex wants to join with */
        let p: Vec<Option<usize>> = self.partitions[level]
            .delta_q
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

        /* pairs of vertices to merge are found where p_i = j and p_j = i
         * (where vertices want to join with each other)
         */
        let pairs: Vec<(usize, usize)> = p
            .iter()
            .enumerate()
            .filter_map(|(i, j)| match j {
                None => None,
                Some(j) => {
                    if *j < i || p[*j].is_none() {
                        None
                    } else if p[*j].unwrap() != i {
                        None
                    } else {
                        Some((i, *j))
                    }
                }
            })
            .collect();

        /* vector of super vertices - vertices that are grouped together */
        let mut aggregates: Vec<Vec<usize>> =
            vec![vec![]; self.partitions[level].aggregates.len() - pairs.len()];

        /* the size of the coarsened graph */
        let nc = n - pairs.len();

        /* piecewise constant interpolation/prolongation matrix - the coarsening transformation matrix */
        let mut pci = CsMat::zero((n, nc));
        let pairs_len = pairs.len();

        /* group all the pairs and create the transformation matrix */
        for (i, (x, y)) in pairs.iter().enumerate() {
            aggregates[i] = self.partitions[level].aggregates[*x]
                .iter()
                .chain(self.partitions[level].aggregates[*y].iter())
                .map(|z| *z)
                .collect();
            pci.insert(*x, i, 1.0);
            pci.insert(*y, i, 1.0);
        }

        /* filter out all the paired vertices */
        let singles: Vec<usize> = (0..n)
            .filter(|x| !pairs.iter().any(|(i, j)| x == i || x == j))
            .collect();

        /* copy over the non-paired vertex indices */
        for (j, i) in singles.iter().enumerate() {
            aggregates[j + pairs_len] = self.partitions[level].aggregates[*i]
                .iter()
                .map(|z| *z)
                .collect();
            pci.insert(*i, j + pairs_len, 1.0);
        }

        /* to new coarsened graph */
        let mut graph = CsMat::zero((nc, nc));

        /* for each pair of aggregates calculate the edge weights */
        (0..nc).into_iter().for_each(|i| {
            (0..nc).into_iter().for_each(|j| {
                let edge_weight = aggregates[i].iter().fold(0, |acc, x| {
                    acc + aggregates[j].iter().fold(0, |acc, y| {
                        match self.partitions[0].graph.get(*x, *y) {
                            Some(w) => acc + w,
                            None => acc,
                        }
                    })
                });
                graph.insert(i, j, edge_weight);
            })
        });

        /* calculate the new delta modularity matrix */
        let qp = &self.partitions[level].delta_q * &pci;
        let delta_q = &pci.transpose_into() * &qp;

        let coarse = Partition {
            aggregates,
            graph,
            delta_q,
        };

        self.partitions.push(coarse);
    }

    /* loads in some text into a graph */
    fn load() -> Self {
        let mut word_map = HashMap::new();
        let mut word_vec = vec![];

        let path = Path::new("./trump.csv");
        let contents = fs::read_to_string(path).unwrap();

        let mut rdr = csv::Reader::from_reader(contents.as_bytes());
        let mut i: usize = 0;

        /* takes each tweet and puts all the words into its own vec */
        let tweets: Vec<Vec<String>> = rdr
            .records()
            .take(1000)
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
                        for y in (x + 1..x + 4).filter(|y| *y < len) {
                            match word_map.get(&tweet[y]) {
                                Some(j) => {
                                    let val;
                                    /* get the current edge value and increment */
                                    match graph.get(*i, *j) {
                                        Some(edge) => val = edge + 1,
                                        None => val = 1,
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

        /* calculates row sums */
        let row_sums: Vec<usize> = graph
            .outer_iterator()
            .map(|row_vec| row_vec.iter().fold(0, |acc, (_, val)| acc + val))
            .collect();

        /* total of all weights to normalize row sums */
        let total: usize = row_sums.iter().sum();

        /* change in modularity matrix */
        let mut delta_q = CsMat::zero((n, n));

        /* generate change in modularity matrix */
        graph.outer_iterator().enumerate().for_each(|(i, row_vec)| {
            row_vec.iter().for_each(|(j, data)| {
                let b: f64 =
                    *data as f64 - (row_sums[i] as f64 * row_sums[j] as f64 / total as f64);
                let delta_q_ij = 2.0 * b / total as f64;
                delta_q.insert(i, j, delta_q_ij);
            })
        });

        /* first group of aggregates has one vertex per aggregate */
        let aggregates: Vec<Vec<usize>> = (0..n).into_iter().map(|i| vec![i]).collect();

        /* unprocessed graph */
        let finest = Partition {
            aggregates,
            graph,
            delta_q,
        };

        let partitions = vec![finest];

        Data {
            word_vec,
            partitions,
        }
    }
}
