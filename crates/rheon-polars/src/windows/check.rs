//! A core's own invariants, held as a state is read back
//! ([`Windows::check`]), and the damages its loader's tests make: apart
//! from `windows.rs`, under the source cap.

use super::*;

impl Windows {
    /// Whether a core read back from a state file holds what every method
    /// here indexes by (review 2026-10-06, PD1): the loader compared the
    /// kernels, the operators and the clock with its own call's and the
    /// held rows' count with the frame's, and took the rest on trust, so a
    /// damaged file that still decoded panicked at the next call. The rest:
    /// each kernel's operators and the output count are the operators',
    /// the held values `n_outputs` a held row, the rows ready among those
    /// held, each group at its kernels' widths with its waiting rows held
    /// and still open by exactly the windows left to close over them, the
    /// silent list a list of the groups, and the keys the groups one to
    /// one. `Err` names the first that fails.
    ///
    /// # Errors
    ///
    /// A core that is none this module could have built and fed.
    pub fn check(&self) -> Result<(), String> {
        for k in self.kernels.iter() {
            k.check()?;
        }
        let (nk, no) = (self.kernels.len(), self.ops.len());
        let mut members = vec![Vec::new(); nk];
        for (o, op) in self.ops.iter().enumerate() {
            match members.get_mut(op.kernel) {
                Some(m) => m.push(o),
                None => return Err(format!("operator {o} names kernel {}", op.kernel)),
            }
            check_op(o, op, &self.kernels[op.kernel])?;
        }
        let nc = self.cross.as_ref().map_or(0, |c| c.defs().len());
        let np = self.profiles.len();
        if no + nc + np == 0 || members != *self.members || members.iter().any(Vec::is_empty) {
            return Err("its kernels' operators are not its operators".into());
        }
        if let Some(c) = &self.cross
            && (nc == 0 || c.first_out != no || !c.has_shape())
        {
            return Err("its cross-group operators do not follow its operators".into());
        }
        for p in self.profiles.iter() {
            p.check()?;
        }
        // The outputs are the window operators', then the cross-group
        // operators', then the profiles'.
        let nout = no + nc + np;
        if self.n_outputs != nout {
            return Err(format!("{} outputs for {nout} operators", self.n_outputs));
        }
        check_moments(&self.ops, &self.moments)?;
        let is_forward = |k: &KernelDef| k.direction == Direction::Forward;
        let forward = self.kernels.iter().filter(|k| is_forward(k)).count();
        if self.n_forward as usize != forward {
            return Err(format!("{} forward kernels of {forward}", self.n_forward));
        }
        let held = self.held_meta.len();
        if held.checked_mul(nout) != Some(self.held_values.len()) {
            return Err(format!(
                "{} values held for {held} rows of {nout} outputs",
                self.held_values.len()
            ));
        }
        let kept = self.held_meta.iter().take(self.ready).filter(|m| !m.drop);
        if self.ready > held || self.ready_kept != kept.count() {
            return Err(format!("{} rows ready of {held} held", self.ready));
        }
        if self.held_first.checked_add(held as u64) != Some(self.next_seq) {
            return Err(format!(
                "{held} rows held from row {} before row {}",
                self.held_first, self.next_seq
            ));
        }
        // Each group at its kernels' widths, and the windows still open over
        // each held row, which `close` and `flush_group` count down.
        let n_wait = u32::from(
            self.kernels.iter().any(KernelDef::waits)
                || self.profiles.iter().any(ProfileDef::waits),
        );
        let is_held = |seq: u64| seq >= self.held_first && seq < self.next_seq;
        let widths = |v: &[Vec<f64>]| {
            v.len() == nk
                && v.iter()
                    .zip(self.members.iter())
                    .all(|(x, m)| x.len() == m.len())
        };
        let mut open = vec![0u32; held];
        let var: Vec<Vec<bool>> = (0..nk).map(|ki| self.var_of(ki)).collect();
        let quant: Vec<Vec<bool>> = (0..nk).map(|ki| self.quant_of(ki)).collect();
        let ext: Vec<_> = (0..nk).map(|ki| self.ext_of(ki)).collect();
        for (gi, g) in self.groups.iter().enumerate() {
            let fits = g.queues.len() == nk
                && g.queues
                    .iter()
                    .zip(var.iter().zip(&quant).zip(&ext))
                    .all(|(q, ((v, qu), (e, later)))| q.has_width(v, qu, e, *later))
                && g.open.len() == nk
                && widths(&g.open_x)
                && widths(&g.open_held)
                && widths(&g.stamp_values)
                && g.closed.len() == nk
                && g.closed.iter().all(|&c| c <= g.waiting.len())
                && g.last_valued.len() == no
                && g.held.len() == no
                && g.moments.len() == self.moments.len()
                && g.moments
                    .iter()
                    .zip(self.moments.iter())
                    .all(|(r, d)| *r.model.cfg() == d.cfg)
                && (forward > 0 || g.waiting.is_empty())
                && (n_wait > 0 || g.pending.is_empty())
                && g.waiting.iter().all(|w| is_held(w.seq))
                && g.pending.iter().all(|&s| is_held(s))
                && (!g.linked || g.last_raw.is_some())
                && profile::fits(&self.profiles, &g.profile);
            if !fits {
                return Err(format!("group {gi} does not fit its kernels and held rows"));
            }
            for (i, w) in g.waiting.iter().enumerate() {
                let r = (w.seq - self.held_first) as usize;
                let still = self.kernels.iter().zip(&g.closed);
                open[r] += still.filter(|&(k, &c)| is_forward(k) && i >= c).count() as u32;
            }
            for &s in &g.pending {
                open[(s - self.held_first) as usize] += n_wait;
            }
        }
        if let Some(c) = &self.cross {
            c.check_rows(self.next_seq)?;
        }
        if let Some(first) = self.cross.as_ref().and_then(|c| c.waiting_from()) {
            if first < self.held_first {
                return Err("its cross-group stamp opened before its held rows".into());
            }
            for o in &mut open[(first - self.held_first) as usize..] {
                *o += 1;
            }
        }
        if self.held_meta.iter().zip(&open).any(|(m, &o)| m.open != o) {
            return Err("a held row's open windows are not the ones still open over it".into());
        }
        self.check_links()?;
        // The keys name the groups one to one: `group` makes each group
        // under its key, and `push` reads a group by the index a key gives.
        let mut named = vec![false; self.groups.len()];
        for &gi in self.index.values().chain(&self.null_group) {
            match named.get_mut(gi) {
                Some(seen) if !*seen => *seen = true,
                _ => return Err("its keys do not name its groups one to one".into()),
            }
        }
        if named.contains(&false) {
            return Err("its keys do not name its groups one to one".into());
        }
        Ok(())
    }

    /// The silent list: links inside the groups, from the head through each
    /// linked group once, back links that agree, the tail last, and no
    /// unlinked group holding a link ([`Self::check`]).
    fn check_links(&self) -> Result<(), String> {
        let n = self.groups.len();
        let bad = || Err("its silent list is not a list of its groups".to_string());
        let inside = |l: Option<usize>| l.is_none_or(|i| i < n);
        let mut links = self.groups.iter().flat_map(|g| [g.prev_link, g.next_link]);
        if !inside(self.head) || !inside(self.tail) || !links.all(inside) {
            return bad();
        }
        let (mut at, mut prev, mut walked) = (self.head, None, 0);
        while let Some(i) = at {
            let g = &self.groups[i];
            if walked == n || !g.linked || g.prev_link != prev {
                return bad();
            }
            (prev, at, walked) = (Some(i), g.next_link, walked + 1);
        }
        let linked = self.groups.iter().filter(|g| g.linked).count();
        let stray = self
            .groups
            .iter()
            .any(|g| !g.linked && (g.prev_link.is_some() || g.next_link.is_some()));
        if prev != self.tail || walked != linked || stray {
            return bad();
        }
        Ok(())
    }

    /// One named damage, as a bit flip or a truncated array in a state file
    /// would leave the core: for the loader's test in `windows_frame.rs`,
    /// the fields being this module's.
    #[cfg(test)]
    pub(crate) fn damage(&mut self, what: &str) {
        match what {
            "held_values" => {
                self.held_values.pop_back();
            }
            "n_outputs" => self.n_outputs = 0,
            "queue" => self.groups[0].queues[0].n += 1,
            "link" => self.head = Some(self.groups.len()),
            "ready" => self.ready = self.held_meta.len() + 1,
            "waiting" => {
                let past = self.held_first + self.held_meta.len() as u64;
                self.groups[0].waiting[0].seq = past;
            }
            other => unreachable!("no damage {other:?}"),
        }
    }
}
