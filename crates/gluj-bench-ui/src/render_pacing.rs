//! Limit native UI redraws without changing the benchmark worker or its clocks.
use super::MainWindow;
use slint::winit_030::{EventResult, WinitWindowAccessor, winit::event::WindowEvent};
use slint::{ComponentHandle, Timer, TimerMode};
use std::{
    cell::RefCell,
    rc::Rc,
    time::{Duration, Instant},
};

// Rounded up to whole milliseconds: never exceed 60 / 30 frames per second.
const INTERACTIVE_INTERVAL: Duration = Duration::from_millis(17);
const WORKER_BUSY_INTERVAL: Duration = Duration::from_millis(34);

#[derive(Debug, PartialEq)]
enum Decision {
    Render,
    Schedule(Duration),
    Coalesce,
}

#[derive(Default)]
struct FrameGate {
    last_frame: Option<Duration>,
    pending: bool,
}

impl FrameGate {
    fn request(&mut self, now: Duration, interval: Duration) -> Decision {
        let remaining = self
            .last_frame
            .map(|last| interval.saturating_sub(now.saturating_sub(last)))
            .unwrap_or_default();
        if remaining.is_zero() {
            self.last_frame = Some(now);
            self.pending = false;
            Decision::Render
        } else if self.pending {
            Decision::Coalesce
        } else {
            self.pending = true;
            Decision::Schedule(remaining)
        }
    }
    fn cancel_pending(&mut self) {
        self.pending = false;
    }
}

fn timer_delay(remaining: Duration) -> Duration {
    // Slint timers use millisecond precision. Rounding down could create a
    // zero-delay retry before the deadline and spin the event loop.
    Duration::from_millis(remaining.as_nanos().div_ceil(1_000_000) as u64)
}

pub(super) fn install(window: &MainWindow, worker_busy: impl Fn() -> bool + 'static) {
    let gate = Rc::new(RefCell::new(FrameGate::default()));
    let timer = Rc::new(Timer::default());
    let origin = Instant::now();
    let weak_window = window.as_weak();
    window.window().on_winit_window_event(move |_, event| {
        match event {
            WindowEvent::RedrawRequested => {
                let interval = if worker_busy() {
                    WORKER_BUSY_INTERVAL
                } else {
                    INTERACTIVE_INTERVAL
                };
                let decision = gate.borrow_mut().request(origin.elapsed(), interval);
                match decision {
                    Decision::Render => {
                        timer.stop();
                        EventResult::Propagate
                    }
                    Decision::Schedule(remaining) => {
                        let weak_window = weak_window.clone();
                        let weak_gate = Rc::downgrade(&gate);
                        timer.start(TimerMode::SingleShot, timer_delay(remaining), move || {
                            if let Some(gate) = weak_gate.upgrade() {
                                gate.borrow_mut().cancel_pending();
                            }
                            if let Some(window) = weak_window.upgrade() {
                                // Ask the native window directly: Slint still has an outstanding
                                // dirty frame, so its request_redraw() would coalesce this away.
                                window
                                    .window()
                                    .with_winit_window(|native| native.request_redraw());
                            }
                        });
                        EventResult::PreventDefault
                    }
                    Decision::Coalesce => EventResult::PreventDefault,
                }
            }
            WindowEvent::Occluded(true) | WindowEvent::Destroyed => {
                timer.stop();
                gate.borrow_mut().cancel_pending();
                EventResult::Propagate
            }
            // Clicks, keyboard input, hover/leave, resize and close are never
            // delayed or dropped. Only the resulting native paint is paced.
            _ => EventResult::Propagate,
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn redraw_storm_coalesces_and_delivers_the_trailing_frame() {
        let mut gate = FrameGate::default();
        assert_eq!(
            gate.request(Duration::ZERO, INTERACTIVE_INTERVAL),
            Decision::Render
        );
        assert_eq!(
            gate.request(Duration::from_millis(1), INTERACTIVE_INTERVAL),
            Decision::Schedule(Duration::from_millis(16))
        );
        for _ in 0..10_000 {
            assert_eq!(
                gate.request(Duration::from_millis(2), INTERACTIVE_INTERVAL),
                Decision::Coalesce
            );
        }
        assert_eq!(
            gate.request(Duration::from_millis(17), INTERACTIVE_INTERVAL),
            Decision::Render
        );
        assert!(!gate.pending);
    }
    #[test]
    fn frames_stay_under_the_cap_and_busy_transition_obeys_the_longer_budget() {
        for (interval, limit) in [(INTERACTIVE_INTERVAL, 60), (WORKER_BUSY_INTERVAL, 30)] {
            let mut gate = FrameGate::default();
            let frames = (0..1_000_000)
                .filter(|us| gate.request(Duration::from_micros(*us), interval) == Decision::Render)
                .count();
            assert!(frames <= limit);
        }
        let mut gate = FrameGate::default();
        assert_eq!(
            gate.request(Duration::ZERO, INTERACTIVE_INTERVAL),
            Decision::Render
        );
        assert_eq!(
            gate.request(Duration::from_millis(17), WORKER_BUSY_INTERVAL),
            Decision::Schedule(Duration::from_millis(17))
        );
        gate.cancel_pending();
        assert_eq!(
            gate.request(Duration::from_millis(34), WORKER_BUSY_INTERVAL),
            Decision::Render
        );
        // Returning to the interactive rate cancels a longer pending deadline.
        assert_eq!(
            gate.request(Duration::from_millis(35), WORKER_BUSY_INTERVAL),
            Decision::Schedule(Duration::from_millis(33))
        );
        assert_eq!(
            gate.request(Duration::from_millis(51), INTERACTIVE_INTERVAL),
            Decision::Render
        );
    }
    #[test]
    fn submillisecond_deadlines_do_not_schedule_zero_delay_timers() {
        assert_eq!(
            timer_delay(Duration::from_nanos(1)),
            Duration::from_millis(1)
        );
        assert_eq!(
            timer_delay(Duration::from_micros(16_001)),
            Duration::from_millis(17)
        );
    }
}
