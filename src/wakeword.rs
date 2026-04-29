use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct WakewordHit {
    pub keyword: String,
    pub matched_token: String,
    pub confidence: f32,
    pub transcript: String,
}

#[derive(Debug)]
pub struct MariaWakewordSpotter {
    cooldown: Duration,
    last_hit_at: Option<Instant>,
    keyword: String,
    targets: Vec<String>,
}

impl Default for MariaWakewordSpotter {
    fn default() -> Self {
        Self::new_with_targets(
            Duration::from_millis(1500),
            "maria".to_string(),
            vec!["maria".to_string(), "mariam".to_string()],
        )
    }
}

impl MariaWakewordSpotter {
    pub fn new(cooldown: Duration) -> Self {
        Self::default_with_cooldown(cooldown)
    }

    pub fn new_with_targets(cooldown: Duration, keyword: String, variants: Vec<String>) -> Self {
        let targets = build_targets(&keyword, &variants);
        let normalized_keyword = normalize(&keyword)
            .split_whitespace()
            .next()
            .map(str::to_string)
            .unwrap_or_else(|| targets[0].clone());

        Self {
            cooldown,
            last_hit_at: None,
            keyword: normalized_keyword,
            targets,
        }
    }

    fn default_with_cooldown(cooldown: Duration) -> Self {
        Self::new_with_targets(
            cooldown,
            "maria".to_string(),
            vec!["maria".to_string(), "mariam".to_string()],
        )
    }

    pub fn detect(&mut self, transcript: &str) -> Option<WakewordHit> {
        let normalized = normalize(transcript);
        if normalized.is_empty() {
            return None;
        }

        if let Some(last) = self.last_hit_at {
            if last.elapsed() < self.cooldown {
                return None;
            }
        }

        let mut best: Option<(String, f32)> = None;
        for token in normalized.split_whitespace() {
            if token.len() < 5 {
                continue;
            }
            let conf = token_confidence(token, &self.targets);
            if conf < 0.72 {
                continue;
            }

            let replace = match best {
                None => true,
                Some((_, best_conf)) => conf > best_conf,
            };
            if replace {
                best = Some((token.to_string(), conf));
            }
        }

        if let Some((matched_token, confidence)) = best {
            self.last_hit_at = Some(Instant::now());
            return Some(WakewordHit {
                keyword: self.keyword.clone(),
                matched_token,
                confidence,
                transcript: transcript.to_string(),
            });
        }

        None
    }
}

fn token_confidence(token: &str, targets: &[String]) -> f32 {
    let mut best = 0.0f32;
    for target in targets {
        let target = target.as_str();
        let d = levenshtein(token, target);
        let max_len = token.len().max(target.len()) as f32;
        let mut conf = 1.0 - (d as f32 / max_len.max(1.0));

        // Slightly penalize first-letter mismatches to reduce false positives.
        if let Some(target_first) = target.chars().next() {
            if !token.starts_with(target_first) {
                conf -= 0.08;
            }
        } else {
            conf -= 0.08;
        }

        if conf > best {
            best = conf;
        }
    }
    best.clamp(0.0, 1.0)
}

pub type ElektraWakewordSpotter = MariaWakewordSpotter;

fn build_targets(keyword: &str, variants: &[String]) -> Vec<String> {
    let mut out = Vec::new();

    for token in normalize(keyword).split_whitespace() {
        if !token.is_empty() && !out.iter().any(|existing| existing == token) {
            out.push(token.to_string());
        }
    }

    for variant in variants {
        for token in normalize(variant).split_whitespace() {
            if !token.is_empty() && !out.iter().any(|existing| existing == token) {
                out.push(token.to_string());
            }
        }
    }

    if out.is_empty() {
        out.push("maria".to_string());
    }

    out
}

fn normalize(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_space = true;

    for ch in input.chars().flat_map(|c| c.to_lowercase()) {
        if ch.is_ascii_alphabetic() {
            out.push(ch);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }

    out.trim().to_string()
}

fn levenshtein(a: &str, b: &str) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
    let mut curr = vec![0usize; b_chars.len() + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = if ca == *cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1)
                .min(curr[j] + 1)
                .min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[b_chars.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_exact_wakeword() {
        let mut s = MariaWakewordSpotter::new(Duration::from_millis(0));
        let hit = s.detect("Hey Maria, turn on the lights").expect("must detect");
        assert_eq!(hit.keyword, "maria");
        assert!(hit.confidence > 0.90);
    }

    #[test]
    fn detects_common_variant() {
        let mut s = MariaWakewordSpotter::new(Duration::from_millis(0));
        let hit = s.detect("Hey Mariam please respond").expect("must detect variant");
        assert!(hit.confidence > 0.75);
    }

    #[test]
    fn rejects_unrelated_phrase() {
        let mut s = MariaWakewordSpotter::new(Duration::from_millis(0));
        assert!(s.detect("hello assistant what's the weather").is_none());
    }

    #[test]
    fn supports_runtime_keyword_variants() {
        let mut s = MariaWakewordSpotter::new_with_targets(
            Duration::from_millis(0),
            "jarvis".to_string(),
            vec!["jarviss".to_string()],
        );
        let hit = s.detect("hey jarviss play music").expect("must detect runtime variant");
        assert_eq!(hit.keyword, "jarvis");
    }
}