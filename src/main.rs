use csv;
use sprs::CsMat;
use std::{collections::HashMap, fs, path::Path};

fn main() {
    let mut data = Data::load();
    for _ in 0..10 {
        data.coursen();
    }

    data.print_clusters();
    /*
        let mod_sums: Vec<f64> = mod_mat
            .outer_iterator()
            .map(|row_vec| row_vec.iter().fold(0.0, |acc, (_, val)| acc + val))
            .collect();

    */

    //   println!("mod sums: {:?}", mod_sums);

    /*
    let p2: Vec<Option<usize>> = qc1
        .outer_iterator()
        .map(|row_vec| {
            row_vec
                .iter()
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

    let pairs2: Vec<(usize, usize)> = p2
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

    println!("{:?}", pairs2);
    for (x, y) in pairs2 {
    }
    */
}

struct Data {
    tweets: Vec<Vec<String>>,
    word_map: HashMap<String, usize>,
    word_vec: Vec<String>,
    partitions: Vec<Partition>,
}

struct Partition {
    aggregates: Vec<Vec<usize>>,
    graph: CsMat<usize>,
    delta_Q: CsMat<f64>,
    row_sums: Vec<usize>,
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

    fn coursen(&mut self) {
        let level = self.partitions.len() - 1;
        let n = self.partitions[level].aggregates.len();

        let p: Vec<Option<usize>> = self.partitions[level]
            .delta_Q
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

        let mut aggregates: Vec<Vec<usize>> =
            vec![vec![]; self.partitions[level].aggregates.len() - pairs.len()];
        let nc = n - pairs.len();
        let mut P = CsMat::zero((n, nc));
        let pairs_len = pairs.len();

        for (i, (x, y)) in pairs.iter().enumerate() {
            aggregates[i] = self.partitions[level].aggregates[*x]
                .iter()
                .chain(self.partitions[level].aggregates[*y].iter())
                .map(|z| *z)
                .collect();
            P.insert(*x, i, 1.0);
            P.insert(*y, i, 1.0);
        }

        println!("{:?}", pairs);
        /*
        for (x, y) in pairs {
        println!("{:?} - {:?}", data.word_vec[x], data.word_vec[y]);
        }
        */

        let singles: Vec<usize> = (0..n)
            .filter(|x| !pairs.iter().any(|(i, j)| x == i || x == j))
            .collect();

        for (j, i) in singles.iter().enumerate() {
            aggregates[j + pairs_len] = self.partitions[level].aggregates[*i]
                .iter()
                .map(|z| *z)
                .collect();
            P.insert(*i, j + pairs_len, 1.0);
        }

        let mut graph = CsMat::zero((nc, nc));

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

        let row_sums: Vec<usize> = graph
            .outer_iterator()
            .map(|row_vec| row_vec.iter().fold(0, |acc, (_, val)| acc + val))
            .collect();

        let qp = &self.partitions[level].delta_Q * &P;
        let delta_Q = &P.transpose_into() * &qp;

        let coarse = Partition {
            aggregates,
            graph,
            delta_Q,
            row_sums,
        };

        self.partitions.push(coarse);
    }

    fn load() -> Self {
        let mut word_map = HashMap::new();
        let mut word_vec = vec![];

        let path = Path::new("./trump.csv");
        let contents = fs::read_to_string(path).unwrap();

        let mut rdr = csv::Reader::from_reader(contents.as_bytes());
        let mut i: usize = 0;

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

        let n = word_vec.len();
        let mut graph = CsMat::zero((n, n));

        for tweet in tweets.iter() {
            let len = tweet.len();
            for (x, word) in tweet.iter().enumerate() {
                match word_map.get(word) {
                    Some(i) => {
                        for y in (x + 1..x + 10).filter(|y| *y < len) {
                            match word_map.get(&tweet[y]) {
                                Some(j) => {
                                    let val;
                                    match graph.get(*i, *j) {
                                        Some(edge) => val = edge + 1,
                                        None => val = 1,
                                    }
                                    graph.insert(*i, *j, val);
                                    graph.insert(*j, *i, val);
                                }
                                None => println!("{:?} not found", y),
                            }
                        }
                    }
                    None => println!("{:?} not found", x),
                }
            }
        }

        let row_sums: Vec<usize> = graph
            .outer_iterator()
            .map(|row_vec| row_vec.iter().fold(0, |acc, (_, val)| acc + val))
            .collect();

        let total: usize = row_sums.iter().sum();

        let mut delta_Q = CsMat::zero((n, n));

        graph.outer_iterator().enumerate().for_each(|(i, row_vec)| {
            row_vec.iter().for_each(|(j, data)| {
                let b: f64 =
                    *data as f64 - (row_sums[i] as f64 * row_sums[j] as f64 / total as f64);
                let delta_q = 2.0 * b / total as f64;
                delta_Q.insert(i, j, delta_q);
            })
        });

        let aggregates: Vec<Vec<usize>> = (0..n).into_iter().map(|i| vec![i]).collect();

        let finest = Partition {
            aggregates,
            graph,
            delta_Q,
            row_sums,
        };

        let partitions = vec![finest];

        Data {
            tweets,
            word_map,
            word_vec,
            partitions,
        }
    }
}
