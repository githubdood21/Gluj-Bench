use super::{ChartPoint, Color, Metric, ModelRc, format_binary_size};
use slint::Model;

pub(super) fn point(
    bytes: u64,
    metric: &Metric,
    geometry: [f64; 3],
    axis: (&str, f64, &str),
    accent: Color,
) -> ChartPoint {
    let [x, y, bottom] = geometry;
    let (title, scale, unit) = axis;
    let display_value = format!("{:.6}", metric.value / scale);
    let display_value = format!(
        "{} {unit}",
        display_value.trim_end_matches('0').trim_end_matches('.')
    );
    ChartPoint {
        x: x as f32,
        y: y as f32,
        left: 110.0,
        bottom: bottom as f32,
        value_label: format!("{:.3} {unit}", metric.value / scale).into(),
        dataset_label: format_binary_size(bytes as f64).into(),
        series: title.split(" · ").next().unwrap_or(title).into(),
        display_value: display_value.into(),
        accent,
    }
}

pub(super) fn nearest(
    points: ModelRc<ChartPoint>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    source_height: f32,
) -> i32 {
    if ![x, y, width, height, source_height]
        .iter()
        .all(|v| v.is_finite())
        || width <= 0.0
        || height <= 0.0
        || source_height <= 0.0
    {
        return -1;
    }
    points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let dx = point.x * width / 900.0 - x;
            let dy = point.y * height / source_height - y;
            let distance = dx * dx + dy * dy;
            (distance <= 14.0 * 14.0).then_some((index, distance))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index as i32)
        .unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use slint::VecModel;
    #[test]
    fn hover_uses_pixel_distance_after_resizing_and_picks_nearest_series() {
        let points = ModelRc::new(VecModel::from(vec![
            ChartPoint {
                x: 450.0,
                y: 100.0,
                ..Default::default()
            },
            ChartPoint {
                x: 450.0,
                y: 110.0,
                ..Default::default()
            },
        ]));
        assert_eq!(nearest(points.clone(), 225.0, 55.0, 450.0, 210.0, 420.0), 1);
        assert_eq!(nearest(points.clone(), 225.0, 50.0, 450.0, 210.0, 420.0), 0);
        assert_eq!(
            nearest(points.clone(), 250.0, 50.0, 450.0, 210.0, 420.0),
            -1
        );
        assert_eq!(nearest(points.clone(), 225.0, 55.0, 0.0, 0.0, 420.0), -1);
        assert_eq!(nearest(points, f32::NAN, 55.0, 450.0, 210.0, 420.0), -1);
    }
}
