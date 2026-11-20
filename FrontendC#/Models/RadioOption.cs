namespace DiabeticsApp.Models
{
    public class RadioOption<T>
    {
        public T Value { get; set; } = default!;
        public string Text { get; set; } = string.Empty;
    }
}
