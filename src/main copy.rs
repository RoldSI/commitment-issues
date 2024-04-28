use clap::{Arg, Command};
use nalgebra::Vector2;
use opencv::{
    core::{self, Point, Scalar},
    highgui,
    imgproc::{self, put_text, circle, arrowed_line},
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

fn segmentation(
    segmentation_type: &str,
    image: &core::Mat,
    current_position: &mut Vector2<i32>,
    segmentation_model_path: Option<&str>,
) -> Result<core::Mat, Box<dyn std::error::Error>> {
    let iris_color_mask = match segmentation_type {
        "testo" => {
            core::Mat::default() // Default placeholder
        },
        "classic" => {
            core::Mat::default() // Default placeholder
        },
        "onnx" => {
            if let Some(_model_path) = segmentation_model_path {
                // ONNX-based segmentation placeholder
            }
            core::Mat::default() // Placeholder for ONNX-based segmentation
        },
        _ => {
            println("FATAL ARGUMENT ERROR!");
            std::process::exit(1);
        },
    };

    Ok(iris_color_mask)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("COMMITMENT ISSUES")
        .arg(
            Arg::new("segmentation_type")
                .required(true)
                .value_parser(["onnx", "classic", "testo"])
                .help("Choose segmentation type"),
        )
        .arg(
            Arg::new("segmentation_model_path")
                .help("Path to the ONNX model file (optional)"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')  // No value required for this flag
                .help("Enable status/debug output"),
        )
        .get_matches();

    let segmentation_type = matches
        .get_one::<String>("segmentation_type")
        .unwrap();
    let segmentation_model_path = matches
        .get_one::<String>("segmentation_model_path")
        .map(|x| x.as_str());
    let verbose = matches.contains_id("verbose");

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

    let mut capture = VideoCapture::from_file("filename.avi", videoio::CAP_ANY)?;

    while !shared_state.lock().unwrap().quit_program {
        let mut frame = core::Mat::default();
        if !capture.read(&mut frame)? {
            break;
        }

        highgui::imshow("raw-image", &frame)?;

        let mut state = shared_state.lock().unwrap();

        let mut iris_color_mask = segmentation(
            segmentation_type,
            &frame,
            &mut state.current_position,
            segmentation_model_path,
        )?;

        if state.setup {
            let diff_position = state.current_position - state.ground_truth_position;

            let hor_diff_angle = (diff_position[0] as f32) / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_HOR;
            let hor_angle = hor_diff_angle.atan();
            let hor_degree = hor_angle.to_degrees();

            let ver_diff_angle = (diff_position[1] as f32) / EYE_BALL_RADIUS_PIXEL_EQUIVALENT_VER;
            let ver_angle = ver_diff_angle.atan();
            let ver_degree = ver_angle.to_degrees();

            let org = Point::new(50, 50);
            let font = imgproc::FONT_HERSHEY_SIMPLEX;
            let font_scale = 1.0;
            let thickness = 8;

            let color = Scalar::new(0.0, 255.0, 0.0, 0.0);

            let _ = circle(
                &mut iris_color_mask,
                Point::new(state.current_position[0], state.current_position[1]),
                10,
                color,
                -1,
                8,
                0,
            );

            let goal_position = state.current_position + diff_position;

            let _ = arrowed_line(
                &mut iris_color_mask,
                Point::new(state.current_position[0], state.current_position[1]),
                Point::new(goal_position[0], goal_position[1]),
                Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                8,
                0,
                0.2,
            );

            let _ = put_text(
                &mut iris_color_mask,
                &format!(
                    "hor gaze angle is {:.2}° and ver angle is {:.2}°",
                    hor_degree, ver_degree
                ),
                org,
                font,
                font_scale,
                color,
                thickness,
                imgproc::LINE_AA,
                true,
            );

            highgui::imshow("masked-image", &iris_color_mask)?;

            if verbose {
                println!(
                    "FPS: {:.2}",
                    frame_counter as f32 / (Instant::now() - start_time).as_secs_f32()
                );
                frame_counter += 1;
            }
        } else {
            println!("Calibration not set");
        }

        highgui::wait_key(100)?;

        if !verbose {
            continue;
        }
    }

    highgui::destroy_all_windows()?;

    Ok(())
}
