use clap::{Arg, Command};
use nalgebra::Vector2;
use opencv::{
    core::{self, Point, Scalar},
    highgui,
    imgproc::{self, circle, arrowed_line, put_text},
    videoio::{self, VideoCapture, VideoCaptureTrait},
};
use rdev::{Event, EventType, Key, listen};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant};

const EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR: f32 = 246.77;
const EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER: f32 = 213.1;

struct SharedState {
    setup: bool,
    ground_truth_position: Vector2<i32>,
    current_position: Vector2<i32>,
    quit_program: bool,
}

fn handle_key_press(shared_state: Arc<Mutex<SharedState>>) {
    let callback = move |event: Event| {
        let mut state = shared_state.lock().unwrap();
        if let Some(name) = event.name {
            if name == "e" {
                if !state.setup {
                    state.setup = true;
                    state.ground_truth_position = state.current_position;
                    println!("Calibration successful!");
                }
            } else if event.event_type == EventType::KeyPress(Key::Escape) {
                state.quit_program = true;
            }
        }
    };

    thread::spawn(move || {
        if let Err(err) = listen(callback) {
            println!("Error: {:?}", err);
        }
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("COMMITMENT ISSUES")
        .arg(
            Arg::new("verbose")
                .short('v')
                .help("Enable status/debug output"),
        )
        .arg(
            Arg::new("live")
                .help("Use live video feed") // No `takes_value` required
        )
        .get_matches();

    let verbose = matches.contains_id("verbose");
    let live = matches.contains_id("live");

    let shared_state = Arc::new(Mutex::new(SharedState {
        setup: false,
        ground_truth_position: Vector2::new(0, 0),
        current_position: Vector2::new(0, 0),
        quit_program: false,
    }));

    handle_key_press(Arc::clone(&shared_state));

    highgui::named_window("raw-image", highgui::WINDOW_NORMAL)?;
    highgui::named_window("masked-image", highgui::WINDOW_NORMAL)?;

    let mut frame_counter = 0;
    let start_time = Instant::now();

    let mut capture = if live {
        VideoCapture::new(0, videoio::CAP_ANY)?
    } else {
        VideoCapture::from_file("filename.avi", videoio::CAP_ANY)?
    };

    while !shared_state.lock().unwrap().quit_program {
        let mut frame = core::Mat::default();
        if !capture.read(&mut frame)? {
            break;
        }

        highgui::imshow("raw-image", &frame)?;

        let mut state = shared_state.lock().unwrap();

        if state.setup {
            let diff_position = state.current_position - state.ground_truth_position;

            let hor_diff_angle = (diff_position[0] as f32) / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR;
            let hor_degree = hor_diff_angle.atan().to_degrees();

            let ver_diff_angle = (diff_position[1] as f32) / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER;
            let ver_degree = ver_diff_angle.atan().to_degrees();

            circle(
                &mut frame,
                Point::new(state.current_position[0], state.current_position[1]),
                10,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                -1,
                8,
                0,
            );

            let goal_position = state.current_position + diff_position;

            arrowed_line(
                &mut frame,
                Point::new(state.current_position[0], state.current_position[1]),
                Point::new(goal_position[0], goal_position[1]),
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                8,
                0,
                0.2,
            );

            let org = Point::new(50, 50);
            let font = imgproc::FONT_HERSHEY_SIMPLEX;
            let font_scale = 1.0;
            let thickness = 8;

            put_text(
                &mut frame,
                &format!(
                    "hor gaze angle is {:.2}° and ver angle is {:.2}°",
                    hor_degree, ver_degree
                ),
                org,
                font,
                font_scale,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                thickness,
                imgproc::LINE_AA,
                true,
            );

            highgui::imshow("masked-image", &frame)?;

            if verbose {
                println!(
                    "FPS: {:.2}",
                    frame_counter as f32 / (Instant::now() - start_time).as_secs_f32()
                );
                frame_counter += 1;
            }
        } else {
            highgui::imshow("masked-image", &frame)?;
            println!("Calibration not set");
        }

        highgui::wait_key(100)?;
    }

    highgui::destroy_all_windows()?;

    Ok(())
}
